import os
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.mmmfair_env"))

def ensure_openai_key():
    if "OPENAI_API_KEY" not in os.environ:
        print("\n🔐 MMM-Fair requires an OpenAI API key for LLM-powered chat features.")
        print("You can get one at https://platform.openai.com/account/api-keys")
        user_key = input("Please enter your OpenAI API key: ").strip()
        os.environ["OPENAI_API_KEY"] = user_key

        save = input("Would you like to save this key for future use? (y/N): ").strip().lower()
        if save == "y":
            env_path = os.path.expanduser("~/.mmmfair_env")
            with open(env_path, "w") as f:
                f.write(f"OPENAI_API_KEY={user_key}\n")
            print(f"✅ API key saved to {env_path}")
        else:
            print("🔐 Key loaded only for this session.")
    else:
        print("🔑 Found existing OpenAI API key in environment.")

def get_openAI_llm(model="gpt-4o-mini", temperature=0.1):
    llm=None
    try:
        from langchain_community.chat_models import ChatOpenAI 
        #from langchain_openai import OpenAI
        llm = ChatOpenAI(model=model, temperature=temperature)
    except ImportError:
        print("🔕 OpenAI features are disabled. Install with: pip install mmm-fair[llm-gpt]")
    except Exception as e:
        print(f"⚠️ Optional OpenAI setup skipped: {e}")
    return llm