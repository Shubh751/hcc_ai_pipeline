
# HCC pipeline ReadMe

## Steps 1 - Put credentials file and set env path in system enviorment variable
- variable name- GOOGLE_APPLICATION_CREDENTIALS
- value - C:\D_Drive\GenAI\DoctusTech\hcc_ai_pipeline\credentials.json

## Step 2 - Change project name in env file accordingly

## Step 3 - Put input files in data/input directory
	
## Step 4 - Run the project with below command as python script - 
- command > poetry run python -m cli

## Step 5 - To run with Langgraph AI studio
- Run below command from root dir of project
- command > poetry run langgraph dev --config .\src\config\langgraph.json

## Step 6 - To run with Docker
- Run below commands from root dir of project
- command > docker build -t hcc-langgraph .
- command > docker run -p 2024:2024 --name hcc-lg hcc-langgraph
			
# Note - 
- ("gemini-2.5-flash") model is used in code. 
- This code is working with Langgraph AI studio only with direct execution and docker


