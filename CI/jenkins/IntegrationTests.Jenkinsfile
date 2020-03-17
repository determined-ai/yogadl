pipeline {
  agent any
  stages {
    stage('TF2Tests') {
      agent { label 'general' }
      steps {
        sh 'virtualenv --python="$(command -v python3.6)" --no-site-packages venv'
        sh "venv/bin/python -m pip install -r requirements.txt"
        sh "venv/bin/python -m pip install tensorflow==2.1.0"
        sh "venv/bin/python -m pytest tests/integration/local/"
        sh "venv/bin/python -m pytest tests/integration/aws/"
      }
    }
    stage('TF1Tests') {
      agent { label 'general' }
      steps {
        sh 'virtualenv --python="$(command -v python3.6)" --no-site-packages venv'
        sh "venv/bin/python -m pip install -r requirements.txt"
        sh "venv/bin/python -m pip install tensorflow==1.14.0"
        sh "venv/bin/python -m pytest tests/integration/local/"
      }
    }
  }
}
