 from flask import Blueprint, render_template, request, redirect
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import bcrypt
from db_connection import get_db_connection
import traceback

user_bp = Blueprint('user', __name__)
login_manager = LoginManager()

# ✅ 使用者模型
class User(UserMixin):
    def __init__(self, id, phone, password):
        self.id = id
        self.phone = phone
        self.password = password

# ✅ 讓 Flask-Login 載入使用者資料
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, phone, password FROM users WHERE user_id = %s", (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return User(id=result[0], phone=result[1], password=result[2])
    return None

# ✅ 註冊畫面
@user_bp.route('/signup/form')
def user_signup_form():
    return render_template('signup_form.html')

# ✅ 處理註冊 POST
@user_bp.route('/signup', methods=['POST'])
def signup():
    try:
        name = request.form.get('name')
        phone = request.form.get('phone')
        password = request.form.get('password')
        role_id = request.form.get('role_id')

        conn = get_db_connection()
        cursor = conn.cursor()

        # ✅ 檢查手機是否已註冊
        cursor.execute("SELECT user_id FROM users WHERE phone = %s", (phone,))
        existing_user = cursor.fetchone()
        if existing_user:
            conn.close()
            return render_template('signup.html', success=False, message="手機號碼已被註冊")

        # ✅ 加密密碼
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


        # ✅ 寫入使用者資料
        cursor.execute("""
            INSERT INTO users (name, phone, password, role_id, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """, (name, phone, hashed_password, role_id))
        conn.commit()
        conn.close()

        return render_template('signup.html', success=True)
    
    except Exception as e:
        print("⚠️ 註冊錯誤：", e)  # 印出錯誤到後端 console
        traceback.print_exc()  # << 加上這行
        return render_template('signup.html', success=False, message=f"註冊失敗: {str(e)}")

# ✅ 登入畫面
@user_bp.route('/login/form')
def user_login_form():
    return render_template('login_form.html')

# ✅ 處理登入 POST
@user_bp.route('/login', methods=['POST'])
def login():
    try:
        phone = request.form.get('phone')
        password = request.form.get('password')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, phone, password FROM users WHERE phone = %s", (phone,))
        result = cursor.fetchone()
        conn.close()

        if result is None:
            return render_template('login.html', success=False, message="帳號不存在")

        user_id, db_phone, db_password = result
        if bcrypt.checkpw(password.encode('utf-8'), db_password.encode('utf-8')):
            user = User(id=user_id, phone=db_phone, password=db_password)
            login_user(user)
            return redirect('/')
        else:
            return render_template('login.html', success=False, message="密碼錯誤")

    except Exception as e:
        return render_template('login.html', success=False, message=f"登入失敗: {str(e)}")

# ✅ 登出
@user_bp.route('/logout')
@login_required
#只有登入的使用者才能存取
def logout():
    logout_user()
    #清除使用者的登入狀態
    return redirect('/login/form')