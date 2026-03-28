# RUN THIS N AMOUNT OF TIMES
import sys
import time

from status import *
from cache import get_accounts
from config import get_verbose, get_llm_provider, get_nvidia_model
from classes.Tts import TTS
from classes.Twitter import Twitter
from classes.YouTube import YouTube
from llm_provider import select_model
from post_bridge_integration import maybe_crosspost_youtube_short


def _ensure_model(model_arg: str = None):
    """Select the LLM model based on provider config or CLI argument."""
    provider = get_llm_provider()
    if provider == "nvidia_nim":
        select_model(model_arg or get_nvidia_model())
    elif model_arg:
        select_model(model_arg)
    else:
        error("No model specified. Pass model name as third argument or configure nvidia_nim.")
        sys.exit(1)


def run_twitter(account_id: str):
    """Post to a single Twitter account."""
    verbose = get_verbose()
    accounts = get_accounts("twitter")
    for acc in accounts:
        if acc["id"] == account_id:
            if verbose:
                info(f"Initializing Twitter for {acc['nickname']}...")
            twitter = Twitter(acc["id"], acc["nickname"], acc["firefox_profile"], acc["topic"])
            twitter.post()
            if verbose:
                success(f"Done posting for {acc['nickname']}.")
            return True
    error(f"Twitter account {account_id} not found.")
    return False


def run_youtube(account_id: str):
    """Generate and upload video for a single YouTube account."""
    verbose = get_verbose()
    tts = TTS()
    accounts = get_accounts("youtube")
    for acc in accounts:
        if acc["id"] == account_id:
            if verbose:
                info(f"Initializing YouTube for {acc['nickname']}...")
            youtube = YouTube(acc["id"], acc["nickname"], acc["firefox_profile"], acc["niche"], acc["language"])
            youtube.generate_video(tts)
            upload_success = youtube.upload_video()
            if upload_success:
                if verbose:
                    success(f"Uploaded Short for {acc['nickname']}.")
                maybe_crosspost_youtube_short(
                    video_path=youtube.video_path,
                    title=youtube.metadata.get("title", ""),
                    interactive=False,
                )
            else:
                warning(f"YouTube upload failed for {acc['nickname']}.")
            return True
    error(f"YouTube account {account_id} not found.")
    return False


def run_all(platform: str):
    """Run all accounts for a given platform sequentially."""
    verbose = get_verbose()
    accounts = get_accounts(platform)
    if not accounts:
        warning(f"No {platform} accounts found.")
        return

    info(f"Batch: running {len(accounts)} {platform} accounts...")
    for i, acc in enumerate(accounts):
        info(f"  [{i+1}/{len(accounts)}] {acc['nickname']}...")
        try:
            if platform == "twitter":
                run_twitter(acc["id"])
            elif platform == "youtube":
                run_youtube(acc["id"])
            # Brief pause between accounts to avoid rate limits
            if i < len(accounts) - 1:
                time.sleep(10)
        except Exception as e:
            error(f"Failed for {acc['nickname']}: {e}")
            continue
    success(f"Batch complete: {len(accounts)} {platform} accounts processed.")


def main():
    if len(sys.argv) < 2:
        error("Usage: python cron.py <platform> [account_id|all] [model]")
        sys.exit(1)

    purpose = str(sys.argv[1])
    account_id = str(sys.argv[2]) if len(sys.argv) > 2 else "all"
    model_arg = str(sys.argv[3]) if len(sys.argv) > 3 else None

    _ensure_model(model_arg)

    if account_id == "all":
        run_all(purpose)
    elif purpose == "twitter":
        run_twitter(account_id)
    elif purpose == "youtube":
        run_youtube(account_id)
    else:
        error("Invalid purpose. Use 'twitter' or 'youtube'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
