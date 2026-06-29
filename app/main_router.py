from flask import Blueprint, request, redirect, url_for
from flask_login import current_user

# 建立 Blueprint 實例，作為所有端點的進入點
main_bp = Blueprint('main', __name__)

@main_bp.before_request
def require_login():
    allowed_endpoints = ['main.login', 'main.register', 'static']
    if not current_user.is_authenticated and request.endpoint not in allowed_endpoints:
        return redirect(url_for('main.login'))

# 引入所有的子路由設定，這樣它們就會自動註冊到這個 main_bp 上
from .routers import pages, upload, status, training, labels, download, api, auth
