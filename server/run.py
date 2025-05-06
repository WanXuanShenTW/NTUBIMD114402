from app import create_app

app = create_app()

# No changes needed here since it already supports cloud deployment
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)