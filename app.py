from fastapi import FastAPI, HTTPException
from routers import routerPredictions
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(routerPredictions.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes cambiar "*" por ["http://127.0.0.1:5500&quot;] si quieres más seguridad
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

if __name__=="__main__":
    app:run()