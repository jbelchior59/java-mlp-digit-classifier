@echo off
setlocal
cd /d "%~dp0"

where javac >nul 2>nul
if errorlevel 1 (
  echo [ERRO] javac nao foi encontrado. Instala um JDK e volta a tentar.
  exit /b 1
)

if not exist "complete\src\P4.java" (
  echo A reconstruir o pacote original...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$parts=Get-ChildItem 'bundle\part_*' | Sort-Object Name; $b64=''; foreach($p in $parts){$b64 += [IO.File]::ReadAllText($p.FullName)}; [IO.File]::WriteAllBytes('bundle\submission.zip',[Convert]::FromBase64String($b64)); if(Test-Path 'complete'){Remove-Item 'complete' -Recurse -Force}; Expand-Archive -LiteralPath 'bundle\submission.zip' -DestinationPath 'complete' -Force"
  if errorlevel 1 exit /b 1
)

if exist out rmdir /s /q out
mkdir out

pushd complete
javac -encoding UTF-8 -d ..\out src\*.java
if errorlevel 1 (
  popd
  exit /b 1
)

echo.
echo Build concluido. A testar o ensemble com uma entrada tecnica de 400 zeros...
powershell -NoProfile -Command "$x=((1..400 | ForEach-Object {'0'}) -join ' '); $x | java -cp ..\out P4"
if errorlevel 1 (
  popd
  exit /b 1
)
popd

echo.
echo Se apareceu 2 ou 3 acima, o classificador e os 10 modelos foram carregados corretamente.
echo Para classificar dados reais, executa a partir da pasta complete:
echo   java -cp ..\out P4 ^< input.txt
endlocal
