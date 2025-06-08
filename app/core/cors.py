from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings

def setup_cors(app: FastAPI) -> None:
   """
   Configura CORS para permitir requests desde Liferay y otros frontends.
   
   """
   app.add_middleware(
       CORSMiddleware,
       allow_origins=settings.get_cors_origins_list(),      
       allow_credentials=settings.cors_credentials,
       allow_methods=settings.get_cors_methods_list(),      
       allow_headers=settings.get_cors_headers_list(),     
   )