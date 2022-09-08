# 支持三种环境变量： {{ AppFolder }}、{{ WorkspaceFolder }}、{{ RootFolder }}

log = dict(filename=None, level="INFO")


infer = dict(
    MMDetectionInfer=dict(
        type="MMDetectionInfer",
        engine=dict(type="MMDetectionInferEngine",
                    model_file="{{ AppFolder }}/balloon/models/model.pth",
                    config_file="{{ AppFolder }}/balloon/models/yolox_s_8x8_300e_coco.py",
                    device="cuda:0"),
        hooks=[dict(type="DrawBBoxHook",
                    out_dir="{{ WorkspaceFolder }}")],
        confidence_threshold=0.7,
        nms_threshold=0.2,
        max_batch_size=16,
        max_object=100,
        width=640,
        height=640,
        workspace="{{ WorkspaceFolder }}",
        slice=False,
        subsize=640,
        rate=1,
        gap=200,
        padding=True,
        labelnames=("balloon", )))
