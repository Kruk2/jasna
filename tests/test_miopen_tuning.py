from jasna.miopen_tuning import database_name, install, merged, read_entries


def test_database_name_is_the_architecture_and_hex_compute_units():
    assert database_name("gfx1201", 64, ".HIP.fdb.txt") == "gfx120140.HIP.fdb.txt"
    assert database_name("gfx1100", 84, ".db.txt") == "gfx110054.db.txt"


def test_read_entries_skips_lines_without_a_key():
    assert read_entries("a-1=x:1;y:2\n\nnot an entry\nb-2=z:3\n") == {"a-1": "x:1;y:2", "b-2": "z:3"}


def test_merged_replaces_tuned_keys_and_keeps_the_others():
    assert merged("a=old\nb=amd\n", {"a": "tuned", "c": "new"}) == "b=amd\na=tuned\nc=new\n"


def test_merged_is_idempotent():
    once = merged("a=old\nb=amd\n", {"a": "tuned"})
    assert merged(once, {"a": "tuned"}) == once


def test_install_writes_every_known_gpu_of_a_tuned_architecture(tmp_path):
    tuned, system = tmp_path / "tuned", tmp_path / "share" / "miopen" / "db"
    tuned.mkdir()
    (tuned / "gfx1201.HIP.fdb.txt").write_text("conv-a=Solver:1.0\n")
    (tuned / "gfx1201.db.txt").write_text("conv-a=Solver:params\n")
    system.mkdir(parents=True)
    (system / "gfx120140.HIP.fdb.txt").write_text("conv-amd=Other:2.0\n")

    written = install(system, tuned, {"gfx1201": (64, 56), "gfx1100": (96,)})

    assert sorted(path.name for path in written) == [
        "gfx120138.HIP.fdb.txt", "gfx120138.db.txt", "gfx120140.HIP.fdb.txt", "gfx120140.db.txt",
    ]
    assert (system / "gfx120140.HIP.fdb.txt").read_text() == "conv-amd=Other:2.0\nconv-a=Solver:1.0\n"
    assert (system / "gfx120138.db.txt").read_text() == "conv-a=Solver:params\n"
    assert not (system / "gfx110060.HIP.fdb.txt").exists()


def test_install_creates_a_missing_database_folder(tmp_path):
    tuned = tmp_path / "tuned"
    tuned.mkdir()
    (tuned / "gfx1201.HIP.fdb.txt").write_text("conv-a=Solver:1.0\n")
    system = tmp_path / "db"
    install(system, tuned, {"gfx1201": (64,)})
    assert (system / "gfx120140.HIP.fdb.txt").read_text() == "conv-a=Solver:1.0\n"
