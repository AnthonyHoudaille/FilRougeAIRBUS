export IMAGE_FAMILY="tf-latest-gpu" # or "pytorch-latest-cpu" for non-GPU instances
export ZONE="europe-west4-c" # budget: "us-west1-b"
export INSTANCE_NAME="deeplearning-instance"
export INSTANCE_TYPE="n1-highmem-8" # budget: "n1-highmem-4"
export INSTANCE_GPU="type=nvidia-tesla-p4,count=1" #budget: "type=nvidia-tesla-k80,count=1"

# budget: 'type=nvidia-tesla-k80,count=1'
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-p4,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible
		
		
gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080


