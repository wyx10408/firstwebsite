#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个人主页 · 本地小后端
------------------------------------------------------------------
只做两件事：
  1. 把当前目录当静态站点发出去（index.html 等）
  2. 提供一个接口  POST /api/remove-bg  ，把前端的图片转发给 Replicate
     调用 lucataco/remove-bg 模型，把去掉背景的结果图直接回传

为什么需要它？
  Replicate 的 Key 属于服务端机密。如果写在网页 JS 里，任何人按 F12
  都能拿走。所以 Key 只从环境变量 REPLICATE_API_TOKEN 读取，只存在于
  这台机器的这个进程里，永远不下发给浏览器。

运行：
    export REPLICATE_API_TOKEN=r8_xxxxxxxxxxxx     # Windows 用 setx / $env:
    python server.py
    # 然后浏览器打开 http://127.0.0.1:8000

只用 Python 标准库，不需要 pip install 任何东西。
"""

import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============================================================
# 配置
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
TOKEN_ENV = "REPLICATE_API_TOKEN"
MODEL = "lucataco/remove-bg"
API_BASE = "https://api.replicate.com/v1"

PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "127.0.0.1")

MAX_UPLOAD_BYTES = 6 * 1024 * 1024   # 前端已经压过一轮，这里再兜个底
POLL_INTERVAL = 1.0                  # 轮询间隔（秒）
POLL_TIMEOUT = 90                    # 轮询总超时（秒）

# Replicate 前面有 Cloudflare，默认的 Python-urllib UA 会被 403 拦掉
USER_AGENT = "personal-page-remove-bg/1.0 (local script)"

DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=\r\n]+)$")


def log(*args):
    print("[server]", *args, flush=True)


class ToolError(Exception):
    """带 HTTP 状态码的业务错误，会被原样变成 JSON 返回给前端。"""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


# ============================================================
# HTTP 小工具（标准库版 requests）
# ============================================================
def http_request(url, method="GET", body=None, headers=None, timeout=120):
    req = urllib.request.Request(url, data=body, method=method)
    # 小坑记录：Replicate 前面挂着 Cloudflare，urllib 默认的
    # "Python-urllib/3.x" 会被直接挡掉（403，error code 1010）。
    # 所以这里固定带一个自己的 User-Agent。
    req.add_header("User-Agent", USER_AGENT)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()
    except urllib.error.URLError as exc:
        raise ToolError("连接 Replicate 失败：%s" % exc.reason, 502)


def friendly_replicate_error(status, payload):
    """把 Replicate 的错误码翻译成人能看懂的话。"""
    detail = ""
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error") or ""
        if isinstance(detail, list):          # 422 校验错误是一个数组
            detail = json.dumps(detail, ensure_ascii=False)
        detail = str(detail)
    table = {
        401: "REPLICATE_API_TOKEN 无效或已过期，请重新生成一个",
        402: "Replicate 账户余额不足，需要先充值",
        403: "无权访问该模型",
        404: "找不到模型 %s" % MODEL,
        422: "图片不符合模型要求",
        429: "请求太频繁，等几秒再试",
    }
    msg = table.get(status, "Replicate 返回错误（HTTP %s）" % status)
    return msg + ("：" + detail[:300] if detail else "")


# ============================================================
# 核心：调用 Replicate 去背景
# ============================================================
# 小坑记录：lucataco/remove-bg 是 2023 年的老模型，不支持
#   POST /v1/models/{owner}/{name}/predictions   （会返回 404）
# 必须走带版本号的经典接口
#   POST /v1/predictions  {"version": "...", "input": {...}}
# 版本号在运行时从模型信息里读出来，不写死在代码里。
_VERSION_CACHE = {"id": ""}


def resolve_version(token):
    """拿到模型最新版本号（第一次调用后缓存在内存里）。"""
    if _VERSION_CACHE["id"]:
        return _VERSION_CACHE["id"]

    status, _headers, body = http_request(
        "%s/models/%s" % (API_BASE, MODEL),
        headers={"Authorization": "Bearer %s" % token},
        timeout=30,
    )
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        raise ToolError("读取模型信息失败（HTTP %s）" % status, 502)
    if status >= 400:
        raise ToolError(friendly_replicate_error(status, data), 502 if status >= 500 else 400)

    version = ((data.get("latest_version") or {}).get("id") or "").strip()
    if not version:
        raise ToolError("模型 %s 没有可用的版本" % MODEL, 502)
    _VERSION_CACHE["id"] = version
    log("解析到模型版本：%s" % version)
    return version


def create_prediction(token, version, data_uri):
    """发起一次预测。碰到 429 限流会按 retry_after 自动重试一次。"""
    url = "%s/predictions" % API_BASE
    payload = json.dumps({
        "version": version,
        "input": {"image": data_uri},
    }).encode("utf-8")
    headers = {
        "Authorization": "Bearer %s" % token,
        "Content-Type": "application/json",
        "Prefer": "wait",            # 让 Replicate 能同步跑完就同步返回
    }

    for attempt in (1, 2):
        status, _h, body = http_request(url, method="POST", body=payload, headers=headers, timeout=90)
        try:
            pred = json.loads(body.decode("utf-8"))
        except Exception:
            raise ToolError("Replicate 返回了无法解析的内容（HTTP %s）" % status, 502)

        # 账户余额不足 $5 时限流很紧（6 次/分钟、突发 1 次），等一下再试
        if status == 429 and attempt == 1:
            wait = pred.get("retry_after") or 10
            try:
                wait = min(int(wait), 30)
            except Exception:
                wait = 10
            log("被限流了，%d 秒后自动重试" % wait)
            time.sleep(wait)
            continue

        if status >= 400:
            raise ToolError(
                friendly_replicate_error(status, pred),
                502 if status >= 500 or status == 429 else 400,
            )
        return pred

    raise ToolError("请求被限流，稍等一会儿再试", 429)


def run_remove_bg(data_uri):
    token = (os.environ.get(TOKEN_ENV) or "").strip()
    if not token:
        raise ToolError(
            "服务端没有读到环境变量 %s。请先在终端里设置它再重启 server.py。" % TOKEN_ENV,
            500,
        )

    # 1) 拿到版本号 → 发起预测
    version = resolve_version(token)
    pred = create_prediction(token, version, data_uri)

    # 2) 万一 60 秒还没跑完，就轮询到结束
    get_url = (pred.get("urls") or {}).get("get")
    deadline = time.time() + POLL_TIMEOUT
    while pred.get("status") in ("starting", "processing"):
        if not get_url:
            raise ToolError("Replicate 没有返回查询地址，无法继续等待", 502)
        if time.time() > deadline:
            raise ToolError("处理超时（超过 %d 秒），请换张小一点的图片再试" % POLL_TIMEOUT, 504)
        time.sleep(POLL_INTERVAL)
        _, _, body = http_request(
            get_url, headers={"Authorization": "Bearer %s" % token}, timeout=30
        )
        try:
            pred = json.loads(body.decode("utf-8"))
        except Exception:
            raise ToolError("轮询时返回了无法解析的内容", 502)

    if pred.get("status") != "succeeded":
        raise ToolError(
            "模型处理失败：%s" % (pred.get("error") or pred.get("status") or "未知原因"), 502
        )

    # 3) output 可能是字符串，也可能是数组 / 字典，都兼容一下
    output = pred.get("output")
    if isinstance(output, list):
        output = output[0] if output else None
    if isinstance(output, dict):
        output = output.get("url") or output.get("image")
    if not output or not isinstance(output, str):
        raise ToolError("模型没有返回图片结果", 502)

    # 4) 把结果图下载回来，直接回传给浏览器
    #    （同源返回，前端才能稳定地显示 + 下载，也省掉 CORS 的麻烦）
    status, headers, data = http_request(output, timeout=90)
    if status >= 400:
        raise ToolError("下载结果图失败（HTTP %s）" % status, 502)

    content_type = (headers.get("Content-Type") or "").split(";")[0].strip()
    if not content_type.startswith("image/"):
        content_type = "image/png"
    return content_type, data, pred.get("id", "")


# ============================================================
# 请求处理
# ============================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "PersonalPage/1.0"

    # ---------- 小工具 ----------
    def send_json(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, status, content_type, data, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):        # 静音默认日志，自己打
        pass

    # ---------- 路由 ----------
    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/api/health":
            self.send_json(200, {
                "ok": True,
                "model": MODEL,
                "token_configured": bool((os.environ.get(TOKEN_ENV) or "").strip()),
            })
            return

        # 静态文件
        if path == "/":
            path = "/index.html"
        rel = urllib.parse.unquote(path).lstrip("/")
        target = os.path.normpath(os.path.join(ROOT, rel))
        if not target.startswith(ROOT) or not os.path.isfile(target):
            self.send_json(404, {"error": "Not found: %s" % rel})
            return
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(target, "rb") as fh:
            self.send_bytes(200, ctype, fh.read())

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/api/remove-bg":
            self.send_json(404, {"error": "Not found: %s" % path})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ToolError("请求体是空的", 400)
            if length > MAX_UPLOAD_BYTES:
                raise ToolError(
                    "图片太大了（%.1f MB），请换一张小于 %.0f MB 的图"
                    % (length / 1048576.0, MAX_UPLOAD_BYTES / 1048576.0),
                    413,
                )

            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                raise ToolError("请求体不是合法的 JSON", 400)

            data_uri = (payload.get("image") or "").strip()
            match = DATA_URI_RE.match(data_uri)
            if not match:
                raise ToolError("image 字段必须是 base64 data URI（形如 data:image/png;base64,...）", 400)

            try:
                img_bytes = base64.b64decode(match.group(1), validate=False)
            except Exception:
                raise ToolError("图片 base64 解码失败", 400)
            if len(img_bytes) < 64:
                raise ToolError("这看起来不是一张有效的图片", 400)

            log("收到图片 %.1f KB，开始调用 %s ..." % (len(img_bytes) / 1024.0, MODEL))
            started = time.time()
            content_type, data, pred_id = run_remove_bg(data_uri)
            log("完成，用时 %.1fs，输出 %.1f KB" % (time.time() - started, len(data) / 1024.0))

            self.send_bytes(200, content_type, data, {"X-Prediction-Id": pred_id})

        except ToolError as exc:
            log("业务错误：%s" % exc.message)
            self.send_json(exc.status, {"error": exc.message})
        except BrokenPipeError:
            log("前端提前断开了连接")
        except Exception as exc:                       # 兜底，别让服务挂掉
            log("未预期的错误：%r" % exc)
            self.send_json(500, {"error": "服务端内部错误：%s" % exc})


# ============================================================
# 启动
# ============================================================
def main():
    token = (os.environ.get(TOKEN_ENV) or "").strip()
    print()
    print("  个人主页 · 本地服务")
    print("  --------------------------------------------------")
    print("  地址      http://%s:%d" % (HOST, PORT))
    print("  模型      %s" % MODEL)
    if token:
        print("  Token     已读取（%s...%s，长度 %d）" % (token[:6], token[-4:], len(token)))
    else:
        print("  Token     [缺失] 没读到环境变量 %s" % TOKEN_ENV)
        print("            网页前两屏正常，第三屏会提示未配置。")
        print("            Windows : set REPLICATE_API_TOKEN=r8_xxx   然后重开终端")
        print("            macOS/Linux: export REPLICATE_API_TOKEN=r8_xxx")
    print("  --------------------------------------------------")
    print("  按 Ctrl+C 停止")
    print()

    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
