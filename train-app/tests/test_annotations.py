import pytest

from train_app.grounding import join_blocks, parse_grounding, serialize_grounding


def payload(text="<table><tr><td>0012</td><td>−0,25</td></tr></table>"):
    return {
        "markdown": text,
        "blocks": [{"id": "table", "type": "table", "bbox": [10, 20, 900, 800], "content": text}],
        "text_approved": True,
        "geometry_approved": False,
    }


def test_grounding_continuations_and_invalid_boxes_preserve_text():
    text, blocks, warnings = parse_grounding(
        "Вступление\n![text](0,0,100,100)\nПервая часть\n![text+](900,500,20,700)\nПродолжение"
    )
    assert text == "Вступление\n\nПервая часть Продолжение"
    assert blocks[-1]["type"] == "text+" and blocks[-1]["bbox"] is None
    assert warnings
    with pytest.raises(ValueError):
        serialize_grounding(blocks)


def test_edit_is_versioned_and_optimistic(client, page):
    path = f"/api/projects/{page['project_id']}/pages/{page['id']}/annotations"
    first = client.post(path, json=payload())
    assert first.status_code == 201, first.text
    second = client.post(
        path, json={**payload("Исправлено"), "expected_revision": first.json()["id"]}
    )
    assert second.status_code == 201
    assert (
        client.post(
            path, json={**payload("Потерянная правка"), "expected_revision": first.json()["id"]}
        ).status_code
        == 409
    )
    details = client.get(path.removesuffix("/annotations")).json()
    assert len(details["history"]) == 2
    assert details["history"][1]["markdown"] != details["annotation"]["markdown"]


def test_geometry_requires_current_text_and_real_boxes(client, page):
    path = f"/api/projects/{page['project_id']}/pages/{page['id']}/annotations"
    data = payload()
    data["geometry_approved"] = True
    data["blocks"][0]["bbox"] = None
    assert client.post(path, json=data).status_code == 409
    data["blocks"][0]["bbox"] = [10, 10, 100, 100]
    data["text_approved"] = False
    assert client.post(path, json=data).status_code == 409


def test_no_contradicting_page_and_block_text(client, page):
    data = payload()
    data["markdown"] = "Подменённый текст"
    assert (
        client.post(
            f"/api/projects/{page['project_id']}/pages/{page['id']}/annotations", json=data
        ).status_code
        == 409
    )


def test_geometry_round_trip_keeps_table_content():
    source = '![table](10,20,900,950)\n<table><tr><td colspan="2">001,20</td></tr></table>'
    text, blocks, _ = parse_grounding(source)
    assert parse_grounding(serialize_grounding(blocks))[0] == text == join_blocks(blocks)
