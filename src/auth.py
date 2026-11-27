import hashlib
import hmac
import json
import os
from pathlib import Path


class AuthManager:
    def __init__(self, users_path: str | Path = "data/users.json"):
        self.users_path = Path(users_path)
        self.users_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.users_path.exists():
            self._save_users([])

    def _load_users(self) -> list[dict]:
        with self.users_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_users(self, users: list[dict]) -> None:
        with self.users_path.open("w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)

    @staticmethod
    def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
        salt_bytes = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_bytes,
            100_000,
        )
        return salt_bytes.hex(), digest.hex()

    def _next_user_id(self, users: list[dict]) -> int:
        existing_ids = [int(u["user_id"]) for u in users if "user_id" in u]
        return (max(existing_ids) + 1) if existing_ids else 1001

    def register_user(self, username: str, password: str) -> tuple[bool, str, dict | None]:
        clean_name = username.strip()
        if len(clean_name) < 3:
            return False, "Username must be at least 3 characters.", None
        if len(password) < 6:
            return False, "Password must be at least 6 characters.", None

        users = self._load_users()
        if any(u["username"].lower() == clean_name.lower() for u in users):
            return False, "Username already exists.", None

        salt_hex, hash_hex = self._hash_password(password)
        new_user = {
            "user_id": self._next_user_id(users),
            "username": clean_name,
            "salt": salt_hex,
            "password_hash": hash_hex,
        }
        users.append(new_user)
        self._save_users(users)
        return True, "Account created successfully.", {"user_id": new_user["user_id"], "username": clean_name}

    def login_user(self, username: str, password: str) -> tuple[bool, str, dict | None]:
        clean_name = username.strip()
        users = self._load_users()
        matched = next((u for u in users if u["username"].lower() == clean_name.lower()), None)
        if not matched:
            return False, "Invalid username or password.", None

        _, candidate_hash = self._hash_password(password, salt_hex=matched["salt"])
        if not hmac.compare_digest(candidate_hash, matched["password_hash"]):
            return False, "Invalid username or password.", None

        return True, "Logged in successfully.", {"user_id": int(matched["user_id"]), "username": matched["username"]}
