# This file is the entry point for the Gunicorn WSGI server.
# It imports the Flask app instance from the main api.py file.

from api import app

if __name__ == "__main__":
    app.run()
