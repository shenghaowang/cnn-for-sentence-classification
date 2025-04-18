#################################################################################
# Outside container
#################################################################################
image_name := $(shell basename $(CURDIR))
pin-deps:
		docker build -t $(image_name):no-pin --target deps-no-pin .
		docker run --rm -v $(PWD):/project $(image_name):no-pin bash -c "pip freeze > /project/requirements.txt"
		docker image rm $(image_name):no-pin
