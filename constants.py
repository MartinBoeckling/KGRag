from dotenv import load_dotenv
import os
load_dotenv(".secrets")

ai_core_client_id = os.getenv("ai_core_client_id")
ai_core_client_secret = os.getenv("ai_core_client_secret")
ai_core_resource_group = os.getenv("ai_core_resource_group")
ai_core_auth_url = os.getenv("ai_core_auth_url")
ai_core_url = os.getenv("ai_core_url")