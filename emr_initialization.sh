# instruction for setting up EMR
# Step 1: create EMR cluster on AWS web UI --- configure detail
    # Release Label: emr-6.2.0
    # Application: Spark 3.0.1, Zeppelin 0.9.0
# Step 2: establish EMR on VSCode
    # ssh -i ~pem_file_location.pem hadoop@ec2-54-190-136-158.us-west-2.compute.
# Step 3: upload .py file to s3 bucket for faster EMR step
# Step 4: run file as an EMR step
    # add step --> step type: Spark Application --> Application Location: choose .py file
