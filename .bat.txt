@echo off
REM 이 배치파일을 프로젝트 폴더 안에서 실행하면 .vscode/settings.json 자동 생성
mkdir .vscode
echo { "python.defaultInterpreterPath": "C:\\Users\\hhhey\\Documents\\basic_env\\Scripts\\python.exe" } > .vscode\settings.json
