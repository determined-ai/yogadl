pipeline {
  agent any
  stages {
    stage('Check') {
      agent { label 'general' }
      steps {
        sh 'virtualenv --python="$(command -v python3.6)" --no-site-packages venv'
        sh "venv/bin/python -m pip install -r requirements.txt"
        sh "venv/bin/python -m black --check yogadl/ tests/"
        sh "venv/bin/python -m flake8 yogadl/ tests/"
        sh "venv/bin/python -m mypy yogadl/ tests/"
      }
    }
  }
}
