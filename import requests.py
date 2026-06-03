import requests
import json
import csv
import time
import html
import os

#===== User Parameters =====
post_ids = ["43967898", "44296219", "44258193","43207720", "44231126", "44398714","44615909","44466267","44640824","44370675","44306812","44237095","44650249","44447254"]  # 目标帖子ID列表
list_types = [1, 2]  # 1=latest, 2=most popular
page_size = 50       # Number of comments per page (recommended max: 100）
output_folder = r"C:\Users\HP\Documents\fyp chap 4\hoyolab\data(csv)"  # CSV save path

# create folder if not exist
os.makedirs(output_folder, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

# ===== extract comment content (text + emoji) =====
def extract_content(reply):
    rows = reply.get('structured_content_rows', [])
    text = ""
    for row in rows:
        insert = row.get('insert', {})
        t = insert.get('type')
        if t == "ContentTypeText":
            text += insert.get('text', '')
        elif t == "ContentTypeEmoticon":
            emoji_url = insert.get('emoticon', {}).get('url', '')
            text += f"[emoji:{emoji_url}]"
    return html.unescape(text.strip())

# ===== recursive parsing of sub-replies =====
def parse_sub_replies(sub_replies, seen_ids, parent_id=None):
    sub_comments = []
    for sub in sub_replies:
        r = sub.get('reply', {})
        u = sub.get('user', {})
        reply_id = str(r.get('reply_id'))
        if reply_id in seen_ids:
            continue
        seen_ids.add(reply_id)
        content = extract_content(r)
        sub_comments.append({
            "reply_id": reply_id,
            "parent_id": parent_id,
            "user_id": u.get('uid'),
            "nickname": u.get('nickname'),
            "content": content,
            "like_num": sub.get('stat', {}).get('like_num', 0),
            "created_at": r.get('created_at')
        })
        if sub.get('sub_replies'):
            sub_comments.extend(parse_sub_replies(sub['sub_replies'], seen_ids, parent_id=reply_id))
    return sub_comments

# ===== fetch function =====
def fetch_comments(post_id, list_type, page_size=50):
    print(f"\nStart fetching comments for post {post_id} with list_type={list_type}...")
    all_comments = []
    seen_ids = set()
    last_id = "0"
    
    while True:
        url = f"https://bbs-api-os.hoyolab.com/community/post/wapi/getPostReplies?post_id={post_id}&list_type={list_type}&last_id={last_id}&size={page_size}"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"Request failed: {resp.status_code}")
            break

        data = resp.json()
        comments_list = data.get('data', {}).get('list', [])
        if not comments_list:
            print("No more comments, fetching finished")
            break

        new_count = 0
        for item in comments_list:
            r = item.get('reply', {})
            u = item.get('user', {})
            reply_id = str(r.get('reply_id'))
            if reply_id in seen_ids:
                continue
            seen_ids.add(reply_id)
            content = extract_content(r)
            all_comments.append({
                "reply_id": reply_id,
                "parent_id": None,
                "user_id": u.get('uid'),
                "nickname": u.get('nickname'),
                "content": content,
                "like_num": item.get('stat', {}).get('like_num', 0),
                "created_at": r.get('created_at')
            })
            new_count += 1
            if item.get('sub_replies'):
                all_comments.extend(parse_sub_replies(item['sub_replies'], seen_ids, parent_id=reply_id))
        
        print(f"Fetched {new_count} new comments (last_id={last_id})")
        if new_count == 0:
            break
        last_id = str(comments_list[-1]['reply']['reply_id'])
        time.sleep(0.5)
    
    return all_comments

# ===== execute fetching & save CSV =====
for post_id in post_ids:
    post_comments = []
    for lt in list_types:
        post_comments.extend(fetch_comments(post_id, lt, page_size))
    
    output_csv = os.path.join(output_folder, f"hoyolab_post_{post_id}.csv")
    with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['reply_id','parent_id','user_id','nickname','content','like_num','created_at'])
        writer.writeheader()
        writer.writerows(post_comments)
    
    print(f"\nPost {post_id}: total {len(post_comments)} comments fetched")
    print(f"CSV saved to: {output_csv}")