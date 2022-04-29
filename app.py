import asyncio
from aiohttp import web
import infer

routes = web.RouteTableDef()

infers = {"general_image_infer": infer.create_general_image_infer()}


# 服务状态判断
@routes.get('/health')
async def health(request):
    return web.json_response({'START': 'UP'})


# 推理
@routes.post('/generalImage/infer')
async def general_image_infer(request):
    recv = await request.json()  # post json
    infer = infers.get("general_image_infer").bind_event_loop(asyncio.get_running_loop())  # 获取异步推理器并绑定当前事件循环
    if infer is not None:
        futures = infer.forward(recv["SOURCE_IMAGE"])
        results = []
        for future in futures:
            results.append(await future)
        result = {"STATUS": 200, "TASKID": recv.get("TASKID", ""), "LOG": "", "RESULTS": results}
    else:
        result = {"STATUS": 500, "TASKID": recv.get("TASKID", ""), "LOG": ""}
    return web.json_response(result)


async def init(loop, host, port):
    app = web.Application()
    app.add_routes(routes)
    app_runner = web.AppRunner(app)
    await app_runner.setup()
    server = await loop.create_server(app_runner.server, host, port)
    print(f'======== Running on http://{host}:{port} ========')
    return server


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init(loop, host='0.0.0.0', port=11050))
    loop.run_forever()
