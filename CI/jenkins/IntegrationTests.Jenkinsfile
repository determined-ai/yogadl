pipeline {
  agent any
  stages {
    stage('IntegrationTests') {
      agent { label 'general' }
      steps {
        sh 'virtualenv --python="$(command -v python3.6)" --no-site-packages venv'
        sh "venv/bin/python -m pip install -r requirements.txt"
        sh "venv/bin/python -m pytest tests/integration"
      }
    }
  }
}
