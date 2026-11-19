# LinkedIn Profile Optimizer

## Run the app

1. Copy the sample environment file to `.env`:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` and replace `PASTE_YOUR_API_KEY_HERE` with your Gemini API key.
   Keep `GEMINI_MODEL=gemini-3.8-flash` as is. Never upload `.env` to GitHub.

3. Open PowerShell in this project folder and run:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

4. Open the local URL printed in the terminal. Use **Load next shuffled sample**
   to try an example profile.
