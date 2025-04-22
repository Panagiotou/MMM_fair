import os
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.mmmfair_env"))

def ensure_together_key():
    if "TOGETHER_API_KEY" not in os.environ:
        print("\n🔐 MMM-Fair requires an Groq API key for LLM-powered chat features.")
        print("You can get one at https://api.together.ai/")
        user_key = input("Please enter your Together API key: ").strip()
        os.environ["TOGETHER_API_KEY"] = user_key

        save = input("Would you like to save this key for future use? (y/N): ").strip().lower()
        if save == "y":
            env_path = os.path.expanduser("~/.mmmfair_env")
            with open(env_path, "w") as f:
                f.write(f"TOGETHER_API_KEY={user_key}\n")
            print(f"✅ API key saved to {env_path}")
        else:
            print("🔐 Key loaded only for this session.")
    else:
        print("🔑 Found existing Together API key in environment.")

def get_together_llm():
    llm=None
    try:
        from langchain_together import ChatTogether
        llm = ChatTogether(model="black-forest-labs/FLUX.1-schnell-Free", temperature=0.1, api_key=os.environ["TOGETHER_API_KEY"])
    except ImportError:
        print("🔕 TOGETHER features are disabled. Install with: pip install mmm-fair[llm-together]")
    except Exception as e:
        print(f"⚠️ Optional TOGETHER setup skipped: {e}")
    return llm
        