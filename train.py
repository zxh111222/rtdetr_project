from ultralytics.cfg import entrypoint
arg="yolo detect train model=rtdetr-VanillaNet.yaml data=ultralytics/cfg/datasets/crack.yaml"

entrypoint(arg)