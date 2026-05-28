"""Tests for bug fixes and security hardening in PrestaShopClient."""

import pytest

from src.prestashop_mcp.config import Config
from src.prestashop_mcp.prestashop_client import PrestaShopClient


def make_client() -> PrestaShopClient:
    return PrestaShopClient(
        Config(shop_url="https://test-shop.example.com", api_key="test-key")
    )


# ---------------------------------------------------------------------------
# Security: secrets must never be written to logs
# ---------------------------------------------------------------------------

def test_redact_secrets_masks_password():
    client = make_client()
    xml = (
        "<prestashop><customer>"
        "<passwd>SuperSecret123</passwd>"
        "<email>buyer@example.com</email>"
        "</customer></prestashop>"
    )

    redacted = client._redact_secrets(xml)

    assert "SuperSecret123" not in redacted
    assert "buyer@example.com" in redacted  # non-secret data preserved


# ---------------------------------------------------------------------------
# Bug: update_product re-sends read-only fields and breaks the PUT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_product_omits_readonly_fields():
    client = make_client()
    existing = {
        "product": {
            "id": "15",
            "price": "10.00",
            "active": "1",
            "state": "1",
            "id_category_default": "2",
            "name": [{"id": "1", "value": "Old name"}],
            "link_rewrite": [{"id": "1", "value": "old-name"}],
            "description": [{"id": "1", "value": ""}],
            # Read-only / computed fields that the webservice rejects on PUT:
            "quantity": "100",
            "manufacturer_name": "ACME",
            "position_in_category": "3",
        }
    }
    captured = {}

    async def fake(method, endpoint, params=None, data=None):
        if method == "GET":
            return existing
        captured["data"] = data
        return {"product": {"id": "15"}}

    client._make_request = fake

    await client.update_product("15", price=20.0)

    sent = captured["data"]["product"]
    assert "quantity" not in sent
    assert "manufacturer_name" not in sent
    assert "position_in_category" not in sent
    assert sent["price"] == "20.0"


# ---------------------------------------------------------------------------
# Bug: get_shop_info caps lists at limit=1 so counts are always 0/1
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_shop_info_counts_full_resources():
    client = make_client()
    calls = []

    async def fake(method, endpoint, params=None, data=None):
        calls.append((endpoint, params or {}))
        if endpoint == "products":
            return {"products": [{"id": "1"}, {"id": "2"}, {"id": "3"}]}
        if endpoint == "categories":
            return {"categories": [{"id": "1"}, {"id": "2"}]}
        if endpoint == "customers":
            return {"customers": [{"id": "1"}]}
        if endpoint == "orders":
            return {"orders": [{"id": str(i)} for i in range(4)]}
        return {"configurations": []}

    client._make_request = fake

    result = await client.get_shop_info()

    assert result["shop_info"]["product_count"] == 3
    assert result["shop_info"]["category_count"] == 2
    assert result["shop_info"]["customer_count"] == 1
    assert result["shop_info"]["order_count"] == 4

    # The counting queries must request only IDs and must not cap at 1.
    product_calls = [p for (e, p) in calls if e == "products"]
    assert product_calls, "expected at least one products query"
    for p in product_calls:
        assert p.get("limit") != 1, "counting must not cap results at limit=1"
        assert p.get("display") == "[id]"


# ---------------------------------------------------------------------------
# Bug: multilingual fields assume hardcoded language ids 1 and 2
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_languages_uses_active_shop_languages():
    client = make_client()

    async def fake(method, endpoint, params=None, data=None):
        if endpoint == "languages":
            return {"languages": [{"id": "1"}, {"id": "3"}]}
        return {}

    client._make_request = fake

    await client._ensure_languages()

    assert [lang["id"] for lang in client.available_languages] == [1, 3]
    fields = client._init_multilingual_field("hello")
    assert [f["id"] for f in fields] == [1, 3]


# ---------------------------------------------------------------------------
# Bug: install_module fakes success with a POST the webservice ignores
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_install_module_does_not_fake_success():
    client = make_client()
    called = []

    async def fake(method, endpoint, params=None, data=None):
        called.append((method, endpoint))
        return {}

    client._make_request = fake

    result = await client.install_module("ps_testmodule")

    assert "error" in result
    assert not any(m == "POST" for (m, _e) in called), "must not POST a fake install"


# ---------------------------------------------------------------------------
# Bug: clear_cache claims success it cannot guarantee
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clear_cache_reports_its_limitation():
    client = make_client()

    async def fake(method, endpoint, params=None, data=None):
        if endpoint == "configurations":
            return {
                "configurations": [
                    {"id": "1", "name": "PS_CACHE_ENABLED", "value": "1"}
                ]
            }
        return {}

    client._make_request = fake

    result = await client.clear_cache("all")

    assert "note" in result, "result must disclose that cache clear is best-effort"
