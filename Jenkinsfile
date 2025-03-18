pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/Akash-Balaji003/WasteManagement_Backend.git'
            }
        }

        stage('Deploy to EC2') {
            steps {
                sshagent(['your-ec2-ssh-key']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ec2-user@your-ec2-ip << EOF
                        cd /home/ec2-user/backend
                        git pull origin main
                        sudo systemctl restart your-backend-service
                        EOF
                    '''
                }
            }
        }
    }
}