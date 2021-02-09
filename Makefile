dataset = movie#last-fm#yelp2018#synthetic_house#movie
model = EKGCN
config_file = config/$(dataset)_$(model).ini
result_dir = result/$(dataset)/$(model)

log_file = $(dataset)_$(model)_temp.log

time_log = logs/test_time.lprof

run:
	python3 -u src/main.py --dataset $(dataset) --config $(config_file) --result_dir $(result_dir)

server_run:
	nohup make run > logs/$(log_file) 2>&1 &

test_time:
	CUDA_LAUNCH_BLOCKING=1 kernprof -o $(time_log) -l src/main.py --dataset $(dataset) --config $(config_file)

log:
	less logs/$(log_file)

log_time:
	python -m line_profiler $(time_log)

clean:
	rm */.*.swp
