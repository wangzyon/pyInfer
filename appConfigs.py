log_cfg = dict(filename=None, level="INFO")

detection_infer_cfg = dict(
    type="DetectionInfer",
    engine=dict(
        type="MMDetectionInferEngine",
        model_file="C:/Users/wzy/Desktop/async_infer_python/models/balloon/model.pth",
        config_file="C:/Users/wzy/Desktop/async_infer_python/models/balloon/yolox_s_8x8_300e_coco.py",
        device="cuda:0"),
    confidence_threshold=0.7,
    nms_threshold=0.25,
    max_batch_size=16,
    max_object=100,
    width=640,
    height=640,
    plot_result=False,
    workspace="workspace/",
    classes=("balloon",))
