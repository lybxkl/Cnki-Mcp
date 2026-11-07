"""CNKI MCP Server - 知网检索 MCP 服务主入口"""

# 强制禁用输出缓冲，确保日志实时输出到 MCP
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import json
import asyncio
import sys
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor

# 强制 stdout 和 stderr 无缓冲
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True, write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True, write_through=True)

# MCP 的 stdio 通道就是 stdout：任何打到 stdout 的东西都会污染协议流。客户端按行读
# JSON-RPC，读到一行不是 JSON 的就会断开整条连接——实测只要 import 时 config.py 打的
# 那一行 `[CONFIG] ...` 就足以让握手失败（invalid character 'C'）。
#
# 所以把 print 的默认目标改成 stderr：显式写了 file= 的调用不受影响，SDK 自己写协议走的是
# sys.stdout.buffer、不经过 print，协议流照旧。这一段必须在 from .browser import ... 之前，
# 因为 config / browser / AutomaticVerification 里的 print 都发生在导入期或运行时。
import builtins as _builtins

_original_print = _builtins.print


def _print_to_stderr(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    return _original_print(*args, **kwargs)


_builtins.print = _print_to_stderr

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult

from .models import CNKIQueryRequest, InitState
from .browser import get_browser

APP_NAME = "cnki-mcp"
APP_VERSION = "0.1.0"

server = Server(APP_NAME)

# 工具执行超时配置（秒）。调用方（MCP 客户端）对单次调用另有总超时，这里的预算必须
# 明显小于它，否则客户端先断开，使用者只看到「连接失败」而拿不到可执行的原因。
TOOL_TIMEOUTS = {
    "cnki_search": 150,
    "cnki_get_paper_detail": 90,
    "cnki_get_status": 30,
    "cnki_navigate_page": 60,
    "cnki_batch_get_details": 150,
    "cnki_set_page_size": 60,
    "cnki_download_paper": 150,
}

# 浏览器初始化预算（秒）：包含启动浏览器、打开检索页与自动验证
INIT_TIMEOUT_SEC = 120
# 单次工具调用的总预算上限（秒）：必须明显小于 MCP 客户端的调用超时，
# 否则客户端先断开，使用者只看到连接失败而拿不到可执行的原因
TOTAL_BUDGET_CAP_SEC = 150
# 初始化之后必须给实际操作留出的最小预算（秒），否则宁可先返回失败让调用方重试
MIN_OP_BUDGET_SEC = 20


def _tool_deadline(name: str) -> float:
    """本次工具调用的总预算截止点（time.monotonic 基准）
    
    初始化与实际操作共享这一份预算：分开计时会让两段时间叠加后超过客户端总超时。
    总预算 = min(上限, 初始化预算 + 该工具的操作预算)，冷启动时初始化占一部分，
    浏览器已就绪时几乎全部留给操作。
    """
    op_budget = TOOL_TIMEOUTS.get(name, 120)
    return time.monotonic() + min(TOTAL_BUDGET_CAP_SEC, INIT_TIMEOUT_SEC + op_budget)


def _remaining(deadline: float) -> float:
    return max(deadline - time.monotonic(), 0.0)

# 协作式截止时间比外层超时提前的秒数，留给浏览器线程收尾并交还线程
_DEADLINE_MARGIN_SEC = 5
# 超时后恢复浏览器状态的最长等待秒数
_RECOVERY_TIMEOUT_SEC = 20

# Playwright 必须固定在同一个线程中运行，使用单线程池
_browser_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")

# 全局异步锁和事件（用于协调并发初始化）
_init_async_lock = asyncio.Lock()
_init_complete_event = asyncio.Event()


def _set_owner_deadline(func, budget_sec):
    """给持有浏览器对象的调用设置协作式截止时间

    线程池里的任务无法被 asyncio 取消，只能让线程自己发现预算耗尽后返回；
    否则超时的任务会继续占着唯一的 Playwright 线程，把后续调用一起拖到超时。
    """
    owner = getattr(func, "__self__", None)
    if owner is None or not hasattr(owner, "set_deadline"):
        return
    try:
        owner.set_deadline(budget_sec)
    except Exception as e:
        safe_print(f"[DEADLINE] 设置协作式截止时间失败: {e}", file=sys.stderr)


async def _recover_browser_state():
    """在浏览器线程里做一次轻量状态恢复；线程仍被占用时不无限等待"""
    try:
        browser = get_browser()
        if not hasattr(browser, '_reset_on_timeout'):
            return
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(_browser_executor, browser._reset_on_timeout),
            timeout=_RECOVERY_TIMEOUT_SEC
        )
    except Exception as e:
        safe_print(f"[RECOVERY_ERROR] 恢复失败: {e}", file=sys.stderr)


def _invoke_with_deadline(func, budget_sec, args):
    """在浏览器线程内执行函数，并在真正开始执行时才设置协作式截止时间
    
    截止时间存在浏览器对象上，而浏览器线程还负责排队执行其他调用。如果在提交
    任务时设置，后一个排队调用会覆盖前一个正在执行调用的截止时间，协作式取消
    就失效了。放到线程内、紧挨着真正执行的位置设置，才能保证每次调用用自己
    的预算。
    """
    _set_owner_deadline(func, budget_sec)
    try:
        return func(*args)
    finally:
        _set_owner_deadline(func, None)


async def run_in_browser_thread(func, *args, timeout_sec=120):
    """在固定的 Playwright 专用线程中执行函数，避免 greenlet 跨线程错误
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        timeout_sec: 超时时间（秒），默认 120s
    
    Returns:
        函数执行结果
    
    Raises:
        TimeoutError: 若操作超时
    """
    loop = asyncio.get_event_loop()
    # 协作式截止时间比外层超时略早，确保线程能在外层判定超时前自行退出
    budget = max(timeout_sec - _DEADLINE_MARGIN_SEC, timeout_sec * 0.5)
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_browser_executor, _invoke_with_deadline, func, budget, args),
            timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        safe_print(f"[TIMEOUT] {func.__name__} 超过 {timeout_sec}s，尝试恢复浏览器状态", file=sys.stderr)
        # 不清除截止时间：线程可能仍在运行，要保留预算压力让它自己退出
        await _recover_browser_state()
        raise TimeoutError(f"操作超时：{func.__name__}（{timeout_sec}s）")


def _error_result(text: str) -> CallToolResult:
    """把失败结果标记为 isError
    
    只返回普通文本时，客户端会把失败当成功结果交给模型，模型只能靠读文案猜；
    标记之后调用方会把这次调用记为失败，模型也能明确知道需要换策略或重试。
    """
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


# 初始化失败时给使用者的可执行提示：说明去哪操作、之后怎么办、为什么不会反复验证
INIT_FAILED_HINT = (
    "浏览器初始化失败或超时，本次调用未执行。若知网要求滑块验证，请在已打开的浏览器窗口中手动完成验证，"
    "然后重新调用本工具（验证结果会保存在本地浏览器资料目录中，后续调用不再重复验证）。"
)


def _budget_hint(name: str) -> str:
    """预算被初始化占满时的提示：说明没执行操作，并告知重试成本很低"""
    return (
        f"本次调用预算（{TOOL_TIMEOUTS.get(name, 120)}s）几乎全部用于浏览器初始化，未执行本次操作。"
        "浏览器已就绪，直接重试即可（重试不再需要初始化）。"
    )


async def _prepare_browser(deadline: float):
    """在总预算内确保浏览器就绪，返回留给实际操作的秒数；就绪失败返回 None"""
    if not await ensure_browser_ready(deadline):
        return None
    return _remaining(deadline)


def safe_print(msg: str, file=None):
    """安全打印函数，确保输出到 stderr 并刷新缓冲区"""
    try:
        if file:
            print(msg, file=file, flush=True)
        else:
            print(msg, flush=True)
        # 同时输出到 stderr 确保 MCP 能捕获
        if file != sys.stderr:
            sys.stderr.write(msg + '\n')
            sys.stderr.flush()
    except UnicodeEncodeError:
        encoded = msg.encode('utf-8', errors='replace').decode('utf-8')
        if file:
            print(encoded, file=file, flush=True)
        else:
            print(encoded, flush=True)
        if file != sys.stderr:
            sys.stderr.write(encoded + '\n')
            sys.stderr.flush()


async def ensure_browser_ready(deadline: float | None = None) -> bool:
    """确保浏览器已初始化（智能等待机制）
    
    Args:
        deadline: 本次工具调用的总预算截止点；初始化只占用其中一部分，
                  剩余预算留给实际操作，避免两段超时叠加超过客户端总超时
    """
    browser = get_browser()

    if browser.is_ready():
        return True
    
    init_state = browser._init_state

    if init_state == InitState.IN_PROGRESS:
        safe_print("检测到初始化正在进行，等待完成...", file=sys.stderr)
        wait_sec = _remaining(deadline) if deadline is not None else INIT_TIMEOUT_SEC + 30
        try:
            await asyncio.wait_for(_init_complete_event.wait(), timeout=max(wait_sec, 1))
            return browser.is_ready()
        except asyncio.TimeoutError:
            safe_print("等待初始化超时", file=sys.stderr)
            return False

    async with _init_async_lock:
        if browser.is_ready():
            return True

        if init_state == InitState.FAILED:
            _init_complete_event.clear()

        # 初始化预算 = min(自身上限, 总预算 - 给实际操作留出的最小预算)
        init_timeout = float(INIT_TIMEOUT_SEC)
        if deadline is not None:
            init_timeout = max(min(init_timeout, _remaining(deadline) - MIN_OP_BUDGET_SEC), 10.0)

        safe_print(f"开始初始化浏览器（预算 {init_timeout:.0f}s）...", file=sys.stderr)
        try:
            result = await run_in_browser_thread(browser.initialize, timeout_sec=init_timeout)
        except Exception:
            # 初始化抛错时也要唤醒等待者，否则其他调用会一直挂到等待超时
            _init_complete_event.set()
            raise

        if result:
            _init_complete_event.set()

        return result


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="cnki_search",
            description="知网文献检索工具。通过关键词在中国知网 (CNKI) 检索学术文献。服务启动时会自动打开浏览器；若知网要求滑块验证，请让用户在浏览器窗口中手动完成一次，之后会话会被复用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "检索关键词"
                    },
                    "search_type": {
                        "type": "string",
                        "description": "检索类型: SU=主题, TI=篇名, AU=作者, KY=关键词, AB=摘要, FT=全文",
                        "default": "SU",
                        "enum": ["SU", "TI", "AU", "KY", "AB", "FT"]
                    },
                    "db_code": {
                        "type": "string",
                        "description": "数据库代码: CJFD=期刊(默认), CDMD=硕博论文, CMFD=会议论文",
                        "default": "CJFD",
                        "enum": ["CJFD", "CDMD", "CMFD"]
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "每页显示条数，支持10/20/50三个选项。设置后会自动调整知网页面的显示数量。",
                        "default": 10,
                        "enum": [10, 20, 50]
                    },
                    "page_num": {
                        "type": "integer",
                        "description": "页码",
                        "default": 1,
                        "minimum": 1
                    },
                    "filter_resource": {
                        "type": "string",
                        "description": "筛选资源类型: DISSERTATION=学位论文, JOURNAL=学术期刊, CONFERENCE=会议, NEWSPAPER=报纸, BOOK=图书, PATENT=专利, STANDARD=标准, ACHIEVEMENTS=成果",
                        "default": ""
                    }
                },
                "required": ["keyword"]
            }
        ),
        Tool(
            name="cnki_get_paper_detail",
            description="获取知网文章详细信息，包括摘要、关键词、引用格式等完整元数据。",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_url": {
                        "type": "string",
                        "description": "文章详情页的URL（从cnki_search的结果中获取）"
                    }
                },
                "required": ["paper_url"]
            }
        ),
        Tool(
            name="cnki_get_status",
            description="获取当前知网页面状态，包括页面类型、搜索结果数、当前页码、总页数等信息。用于Agent了解当前浏览器状态。",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="cnki_navigate_page",
            description="在搜索结果中翻页（上一页/下一页）。注意：知网不支持直接跳转到指定页码，只能逐页导航。",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "翻页操作：next(下一页), prev(上一页)",
                        "enum": ["next", "prev"]
                    }
                },
                "required": ["action"]
            }
        ),
        Tool(
            name="cnki_batch_get_details",
            description="跨页批量获取多篇论文的详细信息。会自动翻页直到达到目标数量或页数限制。建议max_count≤20以避免触发反爬虫。",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_count": {
                        "type": "integer",
                        "description": "最大获取数量（建议≤20）",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "最大翻页数",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                }
            }
        ),
        Tool(
            name="cnki_set_page_size",
            description="设置搜索结果每页显示数量。支持10/20/50三个选项。修改后页面会自动刷新，当前页码保持不变。",
            inputSchema={
                "type": "object",
                "properties": {
                    "page_size": {
                        "type": "integer",
                        "description": "每页显示数量",
                        "enum": [10, 20, 50]
                    }
                },
                "required": ["page_size"]
            }
        ),
        Tool(
            name="cnki_download_paper",
            description="下载知网论文文件（PDF 或 CAJ 格式）。需要先通过 cnki_get_paper_detail 确认论文有下载链接（can_download=true）。实际能否下载取决于账号权限。\n\n重要：调用此工具时必须明确指定 save_dir（保存目录）。若用户未指定，Agent 应自动选择合适目录（如用户桌面 ~/Desktop 或 ~/Downloads），并在调用前告知用户文件将保存到哪里。",
            inputSchema={
                "type": "object",
                "properties": {
                    "paper_url": {
                        "type": "string",
                        "description": "论文详情页 URL（从 cnki_search 或 cnki_get_paper_detail 的结果中获取）"
                    },
                    "fmt": {
                        "type": "string",
                        "description": "下载格式：pdf 或 caj，默认 pdf",
                        "enum": ["pdf", "caj"],
                        "default": "pdf"
                    },
                    "save_dir": {
                        "type": "string",
                        "description": "文件保存目录的绝对路径。必须明确指定，不可省略。用户未指定时 Agent 应自行选择合理目录（如 ~/Downloads 或 ~/Desktop）并告知用户。"
                    }
                },
                "required": ["paper_url", "save_dir"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """工具调用入口
    
    成功时返回文本内容列表；失败时返回带 isError 标记的结果，
    让调用方明确区分「工具执行失败」与「工具返回了内容」。
    """
    if name == "cnki_search":
        try:
            # 智能识别搜索类型
            keyword = arguments.get("keyword", "")
            search_type = arguments.get("search_type")
            
            # 如果用户没有明确指定搜索类型，尝试根据关键词智能判断
            if not search_type:
                # 检查是否包含作者搜索的指示词
                author_indicators = ["作者", "author", "著者", "姓名"]
                if any(indicator in keyword for indicator in author_indicators):
                    search_type = "AU"
                    # 清理关键词，移除指示词
                    for indicator in author_indicators:
                        keyword = keyword.replace(indicator, "").strip()
                    arguments["keyword"] = keyword
                    arguments["search_type"] = search_type
            
            request = CNKIQueryRequest(**arguments)
        except Exception as e:
            return _error_result(f"参数错误：{str(e)}")

        try:
            deadline = _tool_deadline(name)
            # 使用智能等待机制（对 Agent 透明）
            budget = await _prepare_browser(deadline)
            if budget is None:
                return _error_result(INIT_FAILED_HINT)
            if budget < MIN_OP_BUDGET_SEC:
                return _error_result(_budget_hint(name))
            
            browser = get_browser()
            result = await run_in_browser_thread(browser.search, request, timeout_sec=budget)
            
            result_dict = result.model_dump()
            result_text = json.dumps(result_dict, ensure_ascii=False, indent=2)
            
            return [TextContent(type="text", text=result_text)]
            
        except TimeoutError as e:
            return _error_result(
                f"检索超时（{TOOL_TIMEOUTS[name]}s），本次调用已终止并释放浏览器线程，可直接重试。\n错误：{str(e)}"
            )
        except Exception as e:
            return _error_result(f"检索失败：{str(e)}")
    
    elif name == "cnki_get_paper_detail":
        try:
            paper_url = arguments.get("paper_url")
            if not paper_url:
                return _error_result("错误：缺少 paper_url 参数")
            
            deadline = _tool_deadline(name)
            # 使用智能等待机制
            budget = await _prepare_browser(deadline)
            if budget is None:
                return _error_result(INIT_FAILED_HINT)
            if budget < MIN_OP_BUDGET_SEC:
                return _error_result(_budget_hint(name))
            
            browser = get_browser()
            paper = await run_in_browser_thread(browser.get_paper_detail, paper_url, timeout_sec=budget)
            
            if paper:
                paper_dict = paper.model_dump()
                result_text = json.dumps(paper_dict, ensure_ascii=False, indent=2)
                return [TextContent(type="text", text=result_text)]
            else:
                return _error_result("获取文章详情失败：页面未解析到有效内容，请确认链接来自检索结果")
            
        except TimeoutError as e:
            return _error_result(
                f"获取详情超时（{TOOL_TIMEOUTS[name]}s），本次调用已终止并释放浏览器线程，可直接重试。\n错误：{str(e)}"
            )
        except Exception as e:
            return _error_result(f"获取文章详情失败：{str(e)}")
    
    elif name == "cnki_get_status":
        try:
            deadline = _tool_deadline(name)
            budget = await _prepare_browser(deadline)
            if budget is None:
                return _error_result(INIT_FAILED_HINT)

            browser = get_browser()
            status = await run_in_browser_thread(
                browser.get_page_status, timeout_sec=min(budget, TOOL_TIMEOUTS[name])
            )
            result_text = json.dumps(status, ensure_ascii=False, indent=2)
            return [TextContent(type="text", text=result_text)]

        except TimeoutError as e:
            return _error_result(f"获取状态超时（{TOOL_TIMEOUTS[name]}s）。\n错误：{str(e)}")
        except Exception as e:
            return _error_result(f"获取状态失败：{str(e)}")
    
    elif name == "cnki_navigate_page":
        try:
            action = arguments.get("action")
            if not action:
                return _error_result("错误：缺少 action 参数")
            
            deadline = _tool_deadline(name)
            # 使用智能等待机制
            budget = await _prepare_browser(deadline)
            if budget is None:
                return _error_result(INIT_FAILED_HINT)
            if budget < MIN_OP_BUDGET_SEC:
                return _error_result(_budget_hint(name))
            
            browser = get_browser()
            op_timeout = min(budget, TOOL_TIMEOUTS[name])
            
            if action == "next":
                success = await run_in_browser_thread(browser.next_page, timeout_sec=op_timeout)
            elif action == "prev":
                success = await run_in_browser_thread(browser.prev_page, timeout_sec=op_timeout)
            else:
                return _error_result(f"错误：未知操作 {action}，仅支持 next / prev")
            
            if success:
                # 页码信息只在还有预算时补取，避免为了措辞把整次调用拖到超时
                status_budget = min(_remaining(deadline), TOOL_TIMEOUTS["cnki_get_status"])
                if status_budget >= 5:
                    status = await run_in_browser_thread(
                        browser.get_page_status, timeout_sec=status_budget
                    )
                    return [TextContent(
                        type="text",
                        text=f"翻页成功！当前第 {status.get('current_page')} 页，共 {status.get('total_pages')} 页"
                    )]
                return [TextContent(type="text", text="翻页成功")]
            else:
                return _error_result("翻页失败：当前不在检索结果页或已到边界页，请先执行检索")
            
        except TimeoutError as e:
            return _error_result(
                f"翻页超时（{TOOL_TIMEOUTS[name]}s），本次调用已终止并释放浏览器线程，可直接重试。\n错误：{str(e)}"
            )
        except Exception as e:
            return _error_result(f"翻页失败：{str(e)}")
    
    elif name == "cnki_batch_get_details":
        try:
            max_count = arguments.get("max_count", 10)
            max_pages = arguments.get("max_pages", 3)
            
            deadline = _tool_deadline(name)
            # 使用智能等待机制
            budget = await _prepare_browser(deadline)
            if budget is None:
                return _error_result(INIT_FAILED_HINT)
            if budget < MIN_OP_BUDGET_SEC:
                return _error_result(_budget_hint(name))
            
            browser = get_browser()
            papers = await run_in_browser_thread(
                browser.batch_get_details_across_pages,
                max_count,
                max_pages,
                timeout_sec=min(budget, TOOL_TIMEOUTS[name])
            )
            
            if papers:
                result = {
                    "total": len(papers),
                    "papers": [p.model_dump() for p in papers]
                }
                if len(papers) < max_count:
                    # 未达目标数量的原因可能是结果不足、翻页上限或预算用完，措辞不能只归因于超时
                    result["message"] = (
                        f"未达到目标 {max_count} 篇，实际获取 {len(papers)} 篇；"
                        "可能已到结果末尾、翻页上限，或本次调用预算用完，可减少数量或页数后重试"
                    )
                result_text = json.dumps(result, ensure_ascii=False, indent=2)
                return [TextContent(type="text", text=result_text)]
            
            return _error_result("未获取到论文详情，请先执行 cnki_search 再批量获取")
            
        except TimeoutError as e:
            return _error_result(
                f"批量获取超时（{TOOL_TIMEOUTS[name]}s），本次调用已终止并释放浏览器线程，可减少 max_count 后重试。\n错误：{str(e)}"
            )
        except Exception as e:
            return _error_result(f"批量获取失败：{str(e)}")
    
    elif name == "cnki_set_page_size":
        try:
            page_size = arguments.get("page_size")
            if not page_size:
                return _error_result("错误：缺少 page_size 参数")
            
            if page_size not in [10, 20, 50]:
                return _error_result(f"错误：page_size 必须是 10、20 或 50，当前值：{page_size}")
            
            deadline = _tool_deadline(name)
            # 使用智能等待机制
            budget = await _prepare_browser(deadline)
            if budget is None:
                return _error_result(INIT_FAILED_HINT)
            if budget < MIN_OP_BUDGET_SEC:
                return _error_result(_budget_hint(name))
            
            browser = get_browser()
            success = await run_in_browser_thread(
                browser.set_page_size, page_size, timeout_sec=min(budget, TOOL_TIMEOUTS[name])
            )
            
            if success:
                status_budget = min(_remaining(deadline), TOOL_TIMEOUTS["cnki_get_status"])
                if status_budget >= 5:
                    status = await run_in_browser_thread(
                        browser.get_page_status, timeout_sec=status_budget
                    )
                    return [TextContent(
                        type="text",
                        text=f"成功设置每页显示 {page_size} 条！当前第 {status.get('current_page', 1)} 页，共 {status.get('total_pages', 1)} 页"
                    )]
                return [TextContent(type="text", text=f"成功设置每页显示 {page_size} 条")]
            else:
                return _error_result("设置每页显示数量失败：当前不在检索结果页或页面控件未出现，请先执行 cnki_search")
            
        except TimeoutError as e:
            return _error_result(
                f"设置超时（{TOOL_TIMEOUTS[name]}s），本次调用已终止并释放浏览器线程，可直接重试。\n错误：{str(e)}"
            )
        except Exception as e:
            return _error_result(f"设置失败：{str(e)}")
    
    elif name == "cnki_download_paper":
        try:
            paper_url = arguments.get("paper_url")
            if not paper_url:
                return _error_result("错误：缺少 paper_url 参数")
            
            fmt = arguments.get("fmt", "pdf")
            save_dir = arguments.get("save_dir", "").strip()
            
            # save_dir 为必填项，未传时拒绝执行
            if not save_dir:
                return _error_result(
                    "错误：必须指定 save_dir（文件保存目录）。\n"
                    "请明确告知用户文件将保存到哪里，然后重新调用并传入 save_dir 参数。\n"
                    "建议目录：\n"
                    "  Windows: C:/Users/<用户名>/Downloads 或 C:/Users/<用户名>/Desktop\n"
                    "  macOS/Linux: ~/Downloads 或 ~/Desktop"
                )
            
            import os
            # 展开 ~ 为实际用户目录
            save_dir = os.path.expanduser(save_dir)
            save_dir_abs = os.path.abspath(save_dir)
            
            deadline = _tool_deadline(name)
            budget = await _prepare_browser(deadline)
            if budget is None:
                return _error_result(INIT_FAILED_HINT)
            if budget < MIN_OP_BUDGET_SEC:
                return _error_result(_budget_hint(name))
            
            browser = get_browser()
            result = await run_in_browser_thread(
                browser.download_paper, paper_url, fmt, save_dir_abs,
                timeout_sec=min(budget, TOOL_TIMEOUTS[name])
            )
            
            result_dict = result.model_dump()
            
            # 构建人类可读的摘要
            if result.success:
                summary = (
                    f"下载成功！\n"
                    f"  文件名：{result.file_name}\n"
                    f"  保存路径：{result.file_path}\n"
                    f"  格式：{result.format.upper()}"
                )
            else:
                summary = (
                    f"下载失败：{result.message}\n"
                    f"  目标目录：{save_dir_abs}"
                )
            
            result_dict["summary"] = summary
            result_text = json.dumps(result_dict, ensure_ascii=False, indent=2)
            if not result.success:
                return _error_result(result_text)
            return [TextContent(type="text", text=result_text)]
            
        except TimeoutError as e:
            return _error_result(
                f"下载超时（{TOOL_TIMEOUTS[name]}s），本次调用已终止并释放浏览器线程，可直接重试。\n错误：{str(e)}"
            )
        except Exception as e:
            return _error_result(f"下载失败：{str(e)}")
    
    else:
        return _error_result(f"未知工具: {name}")


async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    safe_print("\n" + "=" * 60, file=sys.stderr)
    safe_print("[*] 启动知网检索 MCP 服务", file=sys.stderr)
    safe_print("=" * 60, file=sys.stderr)
    safe_print("\n[*] 浏览器将在首次调用工具时自动初始化", file=sys.stderr)
    safe_print("[*] 多个并发调用会自动等待初始化完成\n", file=sys.stderr)
    
    asyncio.run(run_server())
