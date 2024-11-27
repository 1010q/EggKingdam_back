from fastapi import Depends, FastAPI, UploadFile, Form, File, Query, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt
from uuid import uuid4
from dotenv import load_dotenv
from io import BytesIO
import os

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのドメインを許可（セキュリティ上、特定のドメインにすることが推奨されます）
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST などすべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Signin(BaseModel):
    email: str
    password: str

class Signup(BaseModel):
    email: str
    password: str
    username: str
    is_man: bool = None
    age: int = None

class MaterialInput(BaseModel):
    rice_amount: int
    egg_amount: int

class Rating(BaseModel):
    soysauce_amount: float
    rice_amount: int
    egg_amount: int
    rating: int

class PostActionRequest(BaseModel):
    action: str  # "get_post", "add_comment", "toggle_star"
    content: str = None


@app.post("/login")
async def login_user(request: Signin):
    auth_response = supabase.auth.sign_in_with_password({"email": request.email,"password": request.password,})
    user_id = auth_response.user.id
    response = supabase.table("profile").select("*").eq("user_id", user_id).execute()
    profile_data = response.data[0]

    return {"access_token": auth_response.session.access_token, "token_type": "bearer", "user_id": profile_data["user_id"], "username": profile_data["username"], "image_url": profile_data["image_url"],}


@app.post("/register")
async def register_user(request: Signup):
    sign_up_response = supabase.auth.sign_up({'email': request.email, 'password': request.password})
    sign_in_response = supabase.auth.sign_in_with_password({"email": request.email, "password": request.password})
    user_id = sign_up_response.user.id

    supabase.table("profile").insert({
        "user_id": user_id,
        "username": request.username,
        "image_url": "https://vsmlnrzidfzdmvawficj.supabase.co/storage/v1/object/public/post_image/profile_images/d5c2c64f-eef3-4744-a3cf-caa0ff947749",
        "is_man": request.is_man,
        "age": request.age 
    }).execute()
    access_token = sign_in_response.session.access_token
    return {"access_token": access_token, "token_type": "bearer", "user_id": user_id, "username": request.username, "image_url": "https://vsmlnrzidfzdmvawficj.supabase.co/storage/v1/object/public/post_image/profile_images/d5c2c64f-eef3-4744-a3cf-caa0ff947749", "is_man": request.is_man, "age": request.age}


@app.post("/logout")
async def logout_user(token: str = Depends(oauth2_scheme)):
    supabase.auth.sign_out()

    return {"message": "ログアウト成功"}


@app.post("/material/input/allmodel")
async def get_model( request: MaterialInput, token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    current_user_id = user_response.user.id

    response = supabase.table("allmodel").select("*").execute()
    data = response.data

    X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
    y = np.array([d["soysauce_amount"] for d in data])

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    new_data = np.array([[request.rice_amount, request.egg_amount, 5]])  # ご飯と卵の量、評価(5固定)
    new_data_poly = poly.transform(new_data)
    predicted_soy_sauce = model.predict(new_data_poly)
    soysauce_amount = round(predicted_soy_sauce[0], 1)

    return {"predicted_soy_sauce": soysauce_amount, "rice_amount": request.rice_amount, "egg_amount": request.egg_amount, "model": "allmodel"}


@app.post("/material/input/eachmodel")
async def get_model(request: MaterialInput, token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    user_id = user_response.user.id
    try:
        response = supabase.table("eachmodel").select("*").filter("user_id", "eq", user_id).execute()
        data = response.data
        data_count = len(data)
    except Exception as e:
        data_count = 0

    X = np.array([[180, 72, 4],[200, 66, 5],[180, 72, 6],[240, 66, 5],[180, 60, 8],[180, 72, 4],[50, 60, 5],[50, 66, 5],[50, 72, 5],[500, 60, 5],[500, 66, 5],[500, 72, 5]])
    y = np.array([5.8, 7.0, 7.0, 7.1, 8.0, 5.0, 3.8, 3.9, 4.0, 15.5, 15.8, 16.1])
    if data_count == 0:
        pass

    elif data_count < 8: # データが少なければ、デフォルトデータと重み付きユーザーデータを使用
        user_X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
        user_y = np.array([d["soysauce_amount"] for d in data])

        weight = 3
        user_X_weighted = np.repeat(user_X, weight, axis=0)
        user_y_weighted = np.repeat(user_y, weight, axis=0)

        X = np.vstack((X, user_X_weighted))
        y = np.concatenate((y, user_y_weighted))

    else: # データが十分にあればユーザーデータのみ使用
        sorted_data = sorted(data, key=lambda d: d["timestamp"], reverse=True) # 新しいデータ3件を重みづけ
        recent_data = sorted_data[:3]

        X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
        y = np.array([d["soysauce_amount"] for d in data])

        recent_X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in recent_data])
        recent_y = np.array([d["soysauce_amount"] for d in recent_data])
        weight = 3
        recent_X_weighted = np.repeat(recent_X, weight, axis=0)
        recent_y_weighted = np.repeat(recent_y, weight, axis=0)
        X = np.vstack((X, recent_X_weighted))
        y = np.concatenate((y, recent_y_weighted))

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    new_data = np.array([[request.rice_amount, request.egg_amount, 5]])  # ご飯と卵の量、評価(5固定)
    new_data_poly = poly.transform(new_data)
    predicted_soy_sauce = model.predict(new_data_poly)
    soysauce_amount = round(predicted_soy_sauce[0], 1)

    return {"predicted_soy_sauce": soysauce_amount, "rice_amount": request.rice_amount, "egg_amount": request.egg_amount, "model": "eachmodel"}


@app.post("/user/TKG/rating")
async def rating(request: Rating, token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    current_user_id = user_response.user.id

    (supabase.table("eachmodel").insert({
        "user_id": current_user_id,
        "rice_amount": request.rice_amount,
        "egg_amount": request.egg_amount,
        "soysauce_amount": request.soysauce_amount,
        "rating": request.rating,
    }).execute())

    (supabase.table("allmodel").insert({
        "rice_amount": request.rice_amount,
        "egg_amount": request.egg_amount,
        "soysauce_amount": request.soysauce_amount,
        "rating": request.rating,
    }).execute())

    response = supabase.table("eachmodel").select("*").filter("user_id", "eq", current_user_id).execute()
    data = response.data
    X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
    y = np.array([d["soysauce_amount"] for d in data])

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    rice_range = np.arange(50, 500, 50)  # ご飯の量の範囲（50g単位）
    new_data = np.array([[rice, 66, 5] for rice in rice_range])
    new_data_poly = poly.transform(new_data)
    predicted_soy_sauce = model.predict(new_data_poly)

    plt.figure(figsize=(4, 4))
    plt.plot(rice_range, predicted_soy_sauce, color='y')
    plt.grid(True)

    image_path = f"eachmodel/{uuid4()}.png"
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    file_content = image_stream.read()
    supabase.storage.from_("eachmodel_image").upload(image_path, file_content, {"content-type": "image/png"})

    image_url = f"{SUPABASE_URL}/storage/v1/object/public/eachmodel_image/{image_path}"
    supabase.table("profile").update({"eachmodel_image": image_url}).eq("user_id", current_user_id).execute()

    return {"image_url": image_url}


@app.get("/allmodel_plot")
async def allmodel_plot():
    response = supabase.table("allmodel").select("*").execute()
    data = response.data
    X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
    y = np.array([d["soysauce_amount"] for d in data])

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    rice_range = np.arange(50, 500, 50)  # ご飯の量の範囲（50g単位）
    new_data = np.array([[rice, 66, 5] for rice in rice_range])
    new_data_poly = poly.transform(new_data)
    predicted_soy_sauce = model.predict(new_data_poly)

    plt.figure(figsize=(4, 4))
    plt.plot(rice_range, predicted_soy_sauce, color='y')
    plt.grid(True)

    image_path = f"allmodel/{uuid4()}.png"
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    file_content = image_stream.read()
    supabase.storage.from_("allmodel_image").upload(image_path, file_content, {"content-type": "image/png"})

    image_url = f"{SUPABASE_URL}/storage/v1/object/public/allmodel_image/{image_path}"
    supabase.table("allmodel_image").insert({"image_url": image_url}).execute()

    return {"image_url": image_url}


@app.post("/add/eachmodel")
async def add_eachmodel(request: Rating, token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    current_user_id = user_response.user.id
    response = (supabase.table("eachmodel").insert({
            "user_id": current_user_id,
            "rice_amount": request.rice_amount,
            "egg_amount": request.egg_amount,
            "soysauce_amount": request.soysauce_amount,
            "rating": request.rating,
        }).execute())
    
    response = supabase.table("eachmodel").select("*").filter("user_id", "eq", current_user_id).execute()
    data = response.data
    data_count = len(data)
    if data_count > 5:
        X = np.array([[d["rice_amount"], d["egg_amount"], d["rating"]] for d in data])
        y = np.array([d["soysauce_amount"] for d in data])

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        rice_range = np.arange(50, 500, 50)  # ご飯の量の範囲（50g単位）
        new_data = np.array([[rice, 66, 5] for rice in rice_range])
        new_data_poly = poly.transform(new_data)
        predicted_soy_sauce = model.predict(new_data_poly)

        plt.figure(figsize=(4, 4))
        plt.plot(rice_range, predicted_soy_sauce, color='y')
        plt.grid(True)

        image_path = f"eachmodel/{uuid4()}.png"
        image_stream = BytesIO()
        plt.savefig(image_stream, format='png')
        image_stream.seek(0)
        file_content = image_stream.read()
        supabase.storage.from_("eachmodel_image").upload(image_path, file_content, {"content-type": "image/png"})

        image_url = f"{SUPABASE_URL}/storage/v1/object/public/eachmodel_image/{uuid4()}/{image_path}"
        supabase.table("profile").update({"eachmodel_image": image_url}).eq("user_id", current_user_id).execute()

    return {"message": "評価を保存しました"}


@app.get("/")
async def home(token: str = Depends(oauth2_scheme), sort_by: str = Query("created_at", enum=["created_at", "star_count"])):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    current_user_id = user_response.user.id
    
    user_data = supabase.table("profile").select("*").eq("user_id", current_user_id).execute()
    username = user_data.data[0]["username"]
    image_url = user_data.data[0]["image_url"]
    eachmodel_image = user_data.data[0]["eachmodel_image"]

    response = supabase.table("notifications").select("*").eq("user_id", current_user_id).execute()
    notifications = response.data

    if sort_by == "star_count":
        posts_response = supabase.table("post").select("*").order("star_count", desc=True).execute()
    else:  # デフォルトで作成日時順
        posts_response = supabase.table("post").select("*").order("created_at", desc=True).execute()
    
    posts = posts_response.data

    allmodel_image_data = supabase.table("allmodel_image").select("image_url").order("id", desc=True).limit(1).execute()

    return {
            "user_id": current_user_id, 
            "username": username, 
            "user_image_url": image_url, 
            "posts": posts,
            "notifications": notifications,
            "eachmodel_image": eachmodel_image,
            "allmodel_image": allmodel_image_data.data[0]["image_url"],
        }


@app.get("/profile/{user_id}")
async def get_profile(user_id: str, token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    current_user_id = user_response.user.id

    user_data = supabase.table("profile").select("username", "image_url").eq("user_id", user_id).execute()
    if not user_data.data:
        raise HTTPException(status_code=404, detail="Profile not found")

    username = user_data.data[0]["username"]
    image_url = user_data.data[0]["image_url"]

    posts_response = supabase.table("post").select("*").eq("user_id", user_id).execute()
    posts = posts_response.data

    total_stars = sum(post["star_count"] for post in posts) if posts else 0

    starred_posts_response = supabase.table("post").select("post_id").eq("user_id", user_id).eq("star_count", 1).execute()

    return {
        "posts": posts,
        "username": username,
        "user_image_url": image_url,
        "star_count": total_stars,
        "starred_posts": starred_posts_response.data
    }

@app.patch("/profile/{user_id}")
async def update_profile(user_id: str, username: str = Form(None), user_image: UploadFile = Form(None), is_man: bool = Form(None), age: int = Form(None), token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    current_user_id = user_response.user.id

    if current_user_id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized to update this profile")

    updated_data = {}
    if username:
        updated_data["username"] = username

    if user_image:
        image_path = f"profile_images/{uuid4()}"
        image_content = await user_image.read()
        response = supabase.storage.from_("post_image").upload(image_path, image_content, {"content-type": "image/png"})
        image_url = f"{SUPABASE_URL}/storage/v1/object/public/post_image/{image_path}"
        updated_data["image_url"] = image_url

    if is_man:
        updated_data["is_man"] = is_man

    if age:
        updated_data["age"] = age

    if updated_data:
        supabase.table("profile").update(updated_data).eq("user_id", user_id).execute()

    return {"message": "プロフィールを更新しました", "updated_data": updated_data}


@app.post("/post/create")
async def create_post(title: str = Form(...), description: str = Form(...), image: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    user_id = user_response.user.id
    image_path = f"posts/{uuid4()}"
    file_content = await image.read()
    response = supabase.storage.from_("post_image").upload(image_path, file_content, {"content-type": "image/png"})
    image_url = f"{SUPABASE_URL}/storage/v1/object/public/post_image/{image_path}"

    response = (supabase.table("post").insert({
        "user_id": user_id,
        "title": title,
        "description": description,
        "image_url": image_url
    }).execute())

    post_id = response.data[0]["post_id"]

    return {"post_id": post_id}


@app.post("/postdetail/{post_id}")
async def handle_post_action(post_id: str, request: PostActionRequest, token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=403)
    current_user_id = user_response.user.id

    if request.action == "get_post":
        # 投稿詳細とコメントを取得
        response = supabase.table("post").select("*").eq("post_id", post_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="投稿が見つかりません")

        post_detail = response.data[0]
        comment_response = supabase.table("comments").select("*").eq("post_id", post_id).execute()
        comments = comment_response.data

        return {"post": post_detail, "comments": comments}

    elif request.action == "add_comment":
        # コメントを追加
        if not request.content:
            raise HTTPException(status_code=400, detail="コメント内容が必要です")

        comment_data = {
            "post_id": post_id,
            "user_id": current_user_id,
            "content": request.content,
        }
        insert_response = supabase.table("comments").insert(comment_data).execute()
        added_comment = insert_response.data[0]

        # 通知を作成
        post_response = supabase.table("post").select("user_id").eq("post_id", post_id).execute()
        if not post_response.data:
            raise HTTPException(status_code=404, detail="投稿が見つかりません")
        author_id = post_response.data[0]["user_id"]

        notification_data = {
            "user_id": author_id,
            "type": "あなたの投稿にコメントがつきました",
            "post_id": post_id
        }
        supabase.table("notifications").insert(notification_data).execute()

        return {"comment": added_comment}

    elif request.action == "toggle_star":
        # スターのトグル
        star_entry = supabase.table("stars").select("id").eq("user_id", current_user_id).eq("post_id", post_id).execute()

        post_response = supabase.table("post").select("star_count").eq("post_id", post_id).execute()
        if not post_response.data:
            raise HTTPException(status_code=404, detail="投稿が見つかりません")
        post = post_response.data[0]
        star_count = post["star_count"]
        starred = bool(star_entry.data)

        if starred:
            supabase.table("stars").delete().eq("user_id", current_user_id).eq("post_id", post_id).execute()
            star_count -= 1
        else:
            supabase.table("stars").insert({"user_id": current_user_id, "post_id": post_id}).execute()
            star_count += 1

        supabase.table("post").update({"star_count": star_count}).eq("post_id", post_id).execute()

        return {"star_count": star_count, "starred": starred}