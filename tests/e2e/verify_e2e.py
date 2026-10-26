import httpx
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_e2e_suite() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # 1. Check GET /api/status
        res_status = client.get("/api/status")
        assert res_status.status_code == 200, f"Status failed: {res_status.text}"
        status_data = res_status.json()
        print("[OK] GET /api/status:", json.dumps(status_data))

        # 2. Check GET /api/keys
        res_keys = client.get("/api/keys")
        assert res_keys.status_code == 200, f"Keys failed: {res_keys.text}"
        keys_data = res_keys.json()
        print(f"[OK] GET /api/keys: {len(keys_data)} keys loaded")

        # 3. a) Create new test key POST /api/keys with label="Key-Gamma-Test"
        new_key_payload = {
            "label": "Key-Gamma-Test",
            "key_value": "AQ.Ab8RN6TestGammaKey_abcdef123456"
        }
        res_create = client.post("/api/keys", json=new_key_payload)
        assert res_create.status_code == 201, f"Create key failed: {res_create.text}"
        created_key = res_create.json()
        key_id = created_key["id"]
        print(f"[OK] POST /api/keys created key ID={key_id}, label={created_key['label']}")

        # 4. b) Switch key status (pause/activate PATCH /api/keys/{id})
        res_pause = client.patch(f"/api/keys/{key_id}", json={"is_disabled": True})
        assert res_pause.status_code == 200
        assert res_pause.json()["is_disabled"] is True
        print(f"[OK] PATCH /api/keys/{key_id} paused (is_disabled=True)")

        res_activate = client.patch(f"/api/keys/{key_id}", json={"is_disabled": False})
        assert res_activate.status_code == 200
        assert res_activate.json()["is_disabled"] is False
        print(f"[OK] PATCH /api/keys/{key_id} activated (is_disabled=False)")

        # 5. c) Playground tests with dry_run=True for gemini-2.5-flash and antigravity
        # Check backend endpoint /api/playground/test
        p1 = client.post("/api/playground/test", json={
            "model_id": "gemini-2.5-flash",
            "prompt": "Test rotation prompt 1",
            "action": "generateContent",
            "dry_run": True
        })
        assert p1.status_code == 200, f"Playground test 1 failed: {p1.text}"
        d1 = p1.json()
        print(f"[OK] POST /api/playground/test (gemini-2.5-flash) routed to {d1['selected_key_label']}")

        p2 = client.post("/api/playground/test", json={
            "model_id": "antigravity",
            "prompt": "Test rotation prompt 2",
            "action": "generateContent",
            "dry_run": True
        })
        assert p2.status_code == 200, f"Playground test 2 failed: {p2.text}"
        d2 = p2.json()
        print(f"[OK] POST /api/playground/test (antigravity) routed to {d2['selected_key_label']}")

        # Check frontend UI endpoint /api/playground alias
        p_play = client.post("/api/playground", json={
            "model_id": "gemini-2.5-flash",
            "prompt": "Test playground alias",
            "action": "generateContent",
            "dry_run": True
        })
        assert p_play.status_code == 200, f"/api/playground returned {p_play.status_code}: {p_play.text}"
        print(f"[OK] POST /api/playground returned HTTP {p_play.status_code} successfully")

        # 6. d) Delete test key
        res_del = client.delete(f"/api/keys/{key_id}")
        assert res_del.status_code == 200
        print(f"[OK] DELETE /api/keys/{key_id} successfully deleted")

        # Verify key is gone
        res_keys_after = client.get("/api/keys")
        assert all(k["id"] != key_id for k in res_keys_after.json())
        print(f"[OK] Verified key ID={key_id} removed from active pool")

if __name__ == "__main__":
    test_e2e_suite()
