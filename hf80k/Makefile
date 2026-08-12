# hf80k — FR3 큐브 쌓기 데이터를 만들어 허깅페이스에 올리는 파이프라인
#
# 처음이면 이 순서로 하면 된다.
#   make assets FROM=aidas     자산 90 MB를 가져온다 (한 번만)
#   cp .env.example .env       토큰과 저장소 이름을 채운다
#   make build                 이미지를 굽는다
#   make smoke                 100개로 동작을 확인한다
#   make run                   본 실행. 기본 8만 개
#
# 진행 상황은 make progress, 로그는 make logs, 중단은 make stop이다.
# 모든 변수는 명령줄에서 덮어쓸 수 있다. 예: make run GPU=2 TARGET=20000

SHELL := /bin/bash
.DEFAULT_GOAL := help

HF80K_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
CONTEXT   := $(abspath $(HF80K_DIR)/..)

IMAGE     ?= fr3-hf80k:latest
GPU       ?= 0
NAME      ?= fr3-hf80k-gpu$(GPU)
ENV_FILE  ?= $(HF80K_DIR)/.env
WORK      ?= /data/hf80k/gpu$(GPU)
CACHE     ?= $(HOME)/docker/isaac-sim
FROM      ?=

# 자산 세 가지. 없으면 이미지는 구워지지만 첫 청크에서 죽는다.
ASSETS := $(HF80K_DIR)/assets/fwd_annotated.hdf5 \
          $(HF80K_DIR)/assets/fr3_cube_system_calibration_bundle_v1 \
          $(HF80K_DIR)/assets/fr3_visual_randomization_v1

.PHONY: help assets check build smoke run run-4gpu logs progress state shell stop clean-work aggregate rebuild

help:
	@echo "hf80k — 사용할 수 있는 명령"
	@echo ""
	@echo "  make assets FROM=aidas   자산 90 MB를 가져온다. FROM은 ssh 별칭이나 로컬 경로"
	@echo "  make check               .env와 자산이 갖춰졌는지 본다"
	@echo "  make build               도커 이미지를 굽는다"
	@echo "  make rebuild             캐시를 쓰지 않고 처음부터 굽는다"
	@echo "  make smoke               100개로 동작 확인. 결과는 $(WORK)-smoke 에 쌓인다"
	@echo "  make run                 본 실행. .env의 TARGET_EPISODES 만큼 만든다"
	@echo "  make run-4gpu            GPU 4대에 2만 개씩 나눠 돌린다"
	@echo "  make logs                컨테이너 로그를 따라간다"
	@echo "  make progress            청크 몇 개가 끝났는지 센다"
	@echo "  make state               지금 계획과 진행 상황을 본다"
	@echo "  make shell               컨테이너 안에 들어가 본다"
	@echo "  make stop                컨테이너를 멈춘다"
	@echo "  make aggregate           청크를 하나로 합쳐 올린다 (마지막에 한 번)"
	@echo "  make clean-work          작업 디렉터리를 통째로 지운다. 되돌릴 수 없다"
	@echo ""
	@echo "바꿀 수 있는 값: GPU=$(GPU) IMAGE=$(IMAGE) WORK=$(WORK) ENV_FILE=$(ENV_FILE)"

assets:
	@test -n "$(FROM)" || { echo "FROM을 준다. 예: make assets FROM=aidas"; exit 1; }
	"$(HF80K_DIR)/scripts/fetch_assets.sh" "$(FROM)"

check:
	@fail=0 ; \
	for p in $(ASSETS) ; do \
	  if [ -e "$$p" ] ; then echo "  있음  $$(basename $$p)" ; \
	  else echo "  없음  $$(basename $$p)   -> make assets FROM=..." ; fail=1 ; fi ; \
	done ; \
	if [ -f "$(ENV_FILE)" ] ; then \
	  echo "  있음  $(notdir $(ENV_FILE))" ; \
	  for k in HF_TOKEN HF_REPO_ID ; do \
	    v=$$(grep -E "^$$k=" "$(ENV_FILE)" | cut -d= -f2- ) ; \
	    if [ -n "$$v" ] ; then echo "  채움  $$k" ; \
	    else echo "  비었음 $$k   -> $(ENV_FILE) 를 채운다" ; fail=1 ; fi ; \
	  done ; \
	else echo "  없음  $(notdir $(ENV_FILE))   -> cp .env.example .env" ; fail=1 ; fi ; \
	exit $$fail

build: check
	DOCKER_BUILDKIT=1 docker build -f "$(HF80K_DIR)/docker/Dockerfile" -t "$(IMAGE)" "$(CONTEXT)"

rebuild: check
	DOCKER_BUILDKIT=1 docker build --no-cache -f "$(HF80K_DIR)/docker/Dockerfile" -t "$(IMAGE)" "$(CONTEXT)"

# 스모크 테스트는 본 실행과 작업 디렉터리를 나눈다. 섞이면 이어서 하기가 꼬인다.
smoke: SMOKE_WORK := $(WORK)-smoke
smoke: TARGET ?= 100
smoke: CHUNK  ?= 50
smoke:
	mkdir -p "$(SMOKE_WORK)" "$(CACHE)/cache/kit" "$(CACHE)/cache/ov"
	-docker rm -f $(NAME)-smoke 2>/dev/null
	docker run -d --name $(NAME)-smoke --gpus '"device=$(GPU)"' \
	  --env-file "$(ENV_FILE)" \
	  -e TARGET_EPISODES=$(TARGET) -e CHUNK_SIZE=$(CHUNK) -e CUDA_VISIBLE_DEVICES=0 \
	  -v "$(SMOKE_WORK)":/work \
	  -v "$(CACHE)/cache/kit":/isaac-sim/kit/cache:rw \
	  -v "$(CACHE)/cache/ov":/root/.cache/ov:rw \
	  "$(IMAGE)"
	@echo "시작했다. 로그: make logs NAME=$(NAME)-smoke"

run:
	mkdir -p "$(WORK)" "$(CACHE)/cache/kit" "$(CACHE)/cache/ov"
	-docker rm -f $(NAME) 2>/dev/null
	docker run -d --name $(NAME) --gpus '"device=$(GPU)"' --restart unless-stopped \
	  --env-file "$(ENV_FILE)" \
	  -e CUDA_VISIBLE_DEVICES=0 \
	  $(if $(TARGET),-e TARGET_EPISODES=$(TARGET),) \
	  $(if $(SEED),-e SEED_BASE=$(SEED),) \
	  -v "$(WORK)":/work \
	  -v "$(CACHE)/cache/kit":/isaac-sim/kit/cache:rw \
	  -v "$(CACHE)/cache/ov":/root/.cache/ov:rw \
	  "$(IMAGE)"
	@echo "시작했다. 로그: make logs GPU=$(GPU)"

# GPU 4대에 2만 개씩. 시드를 다르게 줘야 네 대가 같은 데이터를 만들지 않는다.
run-4gpu:
	@for g in 0 1 2 3 ; do \
	  $(MAKE) --no-print-directory run GPU=$$g TARGET=20000 SEED=$$((42000 + g * 10000)) ; \
	done
	@echo "네 대 모두 시작했다. 진행: make progress WORK='/data/hf80k/gpu*'"

logs:
	docker logs -f $(NAME)

progress:
	@echo "완료한 청크:"
	@grep -sh '"status"' $(WORK)/chunks/*/MANIFEST.json 2>/dev/null | sort | uniq -c || echo "  아직 없음"
	@echo "만든 에피소드:"
	@python3 -c "import glob,json,sys; \
f=glob.glob('$(WORK)/chunks/*/MANIFEST.json'); \
n=sum(json.load(open(p)).get('episodes',0) for p in f); \
print(f'  {n}개 (청크 {len(f)}개)')" 2>/dev/null || echo "  아직 없음"

state:
	@cat $(WORK)/state.json 2>/dev/null || echo "$(WORK)/state.json 이 아직 없다"

shell:
	docker run --rm -it --gpus '"device=$(GPU)"' --env-file "$(ENV_FILE)" \
	  -v "$(WORK)":/work "$(IMAGE)" shell

stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

# 청크를 하나로 합쳐 올린다. 본 실행이 다 끝난 뒤 한 번만 한다.
aggregate:
	docker run --rm --env-file "$(ENV_FILE)" -v "$(WORK)":/work \
	  --entrypoint /opt/hf80k/entrypoint.sh "$(IMAGE)" \
	  shell -c "\$$HF80K_PYTHON /opt/hf80k/src/hf_upload.py --mode aggregate --work_dir /work"

clean-work:
	@read -p "$(WORK) 를 통째로 지운다. 되돌릴 수 없다. 계속하려면 yes: " a ; \
	[ "$$a" = "yes" ] && rm -rf "$(WORK)" && echo "지웠다" || echo "그만둔다"
