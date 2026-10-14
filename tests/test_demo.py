import json

from trust_poc.demo import run_demo


async def test_demo_proves_blocked_send_empty_outbox_and_sanitized_audit(tmp_path, capsys):
    result = await run_demo(tmp_path)
    assert [step["trusted"] for step in result["transitions"]] == [True, False, False, False, True]
    assert result["blocked_send"] and result["outbox_count"] == 0
    assert (tmp_path / "outbox.jsonl").read_text() == ""
    audit_text = (tmp_path / "audit.json").read_text()
    audit = json.loads(audit_text)
    assert any(event["enforcement"] == "denied" for event in audit)
    assert "avery@example.test" not in audit_text
    assert "+1-202-555-0147" not in audit_text
    assert "trusted=True->False" in capsys.readouterr().out
    assert json.loads((tmp_path / "summary.json").read_text()) == result
