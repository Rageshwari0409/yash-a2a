from mangum import Mangum
from app import app

# This wraps the FastAPI app to handle Lambda events
handler = Mangum(app)