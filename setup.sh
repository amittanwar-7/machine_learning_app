mkdir -p ~/.streamlit/
echo "
[general]n
email = "amittanwar71@yahoo.in"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml