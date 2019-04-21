import praw
import requests

# Reddit keys
client_id = 'vDn80yQ5BbWEig'
client_secret = 'TDCbXa_9E8k0dX2DUTdrhmMnVls'
username = 'nsfw_sfw_ml'
password = 'waleedwifi'
user_agent = 'mlclassifier'

def NSFW_SFW(subreddit_name):
    reddit = praw.Reddit(client_id=client_id, 
                         client_secret=client_secret, 
                         username=username, 
                         password=password,
                         user_agent=user_agent)
    subreddit = reddit.subreddit(subreddit_name)
    hot_nsfw = subreddit.hot(limit=20)
    scrapped_urls = []
    for nsfw in hot_nsfw:
        scrapped_urls.append(nsfw.url)
    for img in range(len(scrapped_urls)):
        pic = requests.get(scrapped_urls[img])
        IMG_NAME = "Dataset/Train/SFW/" + str(img) + ".png"
        print('Saved {}'.format(IMG_NAME))
        with open(IMG_NAME, "wb") as file:
            file.write(pic.content)

if __name__ == "__main__":
    LIST_SUBREDDIT = ['HighResNSFW', 'GirlsInBeanies', 'cuteface']
    NSFW_SFW(LIST_SUBREDDIT[0])