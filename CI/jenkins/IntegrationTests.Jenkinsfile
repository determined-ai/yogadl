pipeline {
  agent any
  stages {
    stage('IntegrationTests') {
      agent { label 'general' }
      steps {
        sh script: '''
virtualenv --python="$(command -v python3.6)" --no-site-packages venv
. venv/bin/activate
'''
        sh script: '''
venv/bin/python -m pip install -r requirements.txt
'''
        sh script: '''
venv/bin/python -m pytest tests/integration
'''
      }
    }
  }
}