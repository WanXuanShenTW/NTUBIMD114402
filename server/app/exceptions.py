class NotFoundError(Exception):
    """找不到指定資源時使用，例如查無此用戶或資料。
    通常用於 service 層（業務邏輯判斷），有時 dao 查詢也會用。
    """
    pass

class AlreadyExistsError(Exception):
    """當要新增的資源已經存在時使用，例如帳號重複註冊。
    通常用於 service 層（業務邏輯判斷），有時 dao 新增時也會用。
    """
    pass

class ValidationError(Exception):
    """資料格式或內容驗證失敗時使用，例如欄位缺漏或格式錯誤。
    幾乎只用於 service 層（資料驗證）。
    """
    pass

class DatabaseError(Exception):
    """資料庫操作發生錯誤時使用，例如連線失敗或 SQL 執行錯誤。
    主要用於 dao 層（資料庫存取），service 層可捕捉並轉換為更高層錯誤。
    """
    pass

class PermissionDeniedError(Exception):
    """使用者沒有操作權限時使用，例如存取受限資源。
    幾乎只用於 service 層（權限驗證）。
    """
    pass

class AuthenticationError(Exception):
    """使用者驗證失敗時使用，例如登入失敗或 token 無效。
    幾乎只用於 service 層（認證流程）。
    """
    pass