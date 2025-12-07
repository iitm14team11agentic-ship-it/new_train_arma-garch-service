# new_train_arma-garch-service
for model training but this one runs on schedule also the logic is different

server activation:-
nohup uvicorn training_service:app --host 0.0.0.0 --port 8001 &

for a new cronjob entry run this command:-
crontab /my_cron_job.txt
