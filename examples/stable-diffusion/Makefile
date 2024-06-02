
server:
	# python -m model
	docker run --rm --runtime=nvidia -ePORT=8080 -p8080:8080 -v /mnt/disk/vol1:/mnt/models yangwenz/stable-diffusion:v1

build:
	kservehelper build .

.PHONY: server build