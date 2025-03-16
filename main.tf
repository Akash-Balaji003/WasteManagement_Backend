provider "aws" {
  region = "ap-northeast-1"  # Your AWS region
}

# EC2 Instance
resource "aws_instance" "backend_server" {
  # Instance ID will be imported
}

# RDS Database
resource "aws_db_instance" "backend_db" {
  # RDS Instance will be imported
}

# S3 Bucket
resource "aws_s3_bucket" "backend_bucket" {
  # S3 Bucket will be imported
}

# To import existing resources (run these commands):
# terraform init
# terraform import aws_instance.backend_server i-000c521e404f88dc9
# terraform import aws_db_instance.backend_db devops-project-db.cza0amg6av33.ap-northeast-1.rds.amazonaws.com
# terraform import aws_s3_bucket.backend_bucket devops-project-frontend

output "ec2_public_ip" {
  value = aws_instance.backend_server.public_ip
}

output "rds_endpoint" {
  value = aws_db_instance.backend_db.endpoint
}

output "s3_bucket_name" {
  value = aws_s3_bucket.backend_bucket.id
}
