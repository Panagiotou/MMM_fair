import os
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.mmmfair_env"))

def ensure_groq_key():
    if "GROQ_API_KEY" not in os.environ:
        print("\n🔐 MMM-Fair requires an Groq API key for LLM-powered chat features.")
        print("You can get one at https://console.groq.com/keys")
        user_key = input("Please enter your Groq API key: ").strip()
        os.environ["GROQ_API_KEY"] = user_key

        save = input("Would you like to save this key for future use? (y/N): ").strip().lower()
        if save == "y":
            env_path = os.path.expanduser("~/.mmmfair_env")
            with open(env_path, "w") as f:
                f.write(f"GROQ_API_KEY={user_key}\n")
            print(f"✅ API key saved to {env_path}")
        else:
            print("🔐 Key loaded only for this session.")
    else:
        print("🔑 Found existing Groq API key in environment.")

def get_groq_llm():
    llm=None
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model="gemma2-9b-it", temperature=0.1, api_key=os.environ["GROQ_API_KEY"])
    except ImportError:
        print("🔕 GROQ features are disabled. Install with: pip install mmm-fair[llm-groq]")
    except Exception as e:
        print(f"⚠️ Optional GROQ setup skipped: {e}")
    return llm
        


    
    