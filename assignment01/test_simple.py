from LVVTracer import LVVTracer


def changes_locals_once():
    a = 1
    b = "foo"
    c = None


def changes_locals_twice():
    a = 2
    b = "spam"
    c = False

    a = 3
    b += " eggs"
    c = c or "(fallback)"


def changes_to_equal():
    a = 4
    a = 4


def changes_mutable():
    a = []
    a.append(1)
    a.append(2)


def changes_identity_but_not_equality():
    a = list()
    a = list()
    a = list()


def does_not_change_identity_and_equality():
    a = object()
    a = a
    a = a


def changes_a_but_not_b():
    a = 0
    b = "bar"
    a = 5


global_var = 1


def changes_global():
    global global_var
    global_var = 2


def does_not_change_argument(arg=0):
    pass


def changes_argument(arg=0):
    arg += 42


def calls_itself_recursively_six_times(n=5):
    if n:
        calls_itself_recursively_six_times(n-1)


def calls_other_function():
    a = 5
    changes_locals_once()


def test_tracks_defined_locals():
    # Assumption 1: local variables which are merely defined count as a change
    with LVVTracer(target_func="changes_locals_once") as t:
        changes_locals_once()
    assert t.getLVVmap() == {"a": 1, "b": 1, "c": 1}


def test_tracks_not_equal_locals():
    # Assumption 2: local variables which change to something != count as a change
    with LVVTracer(target_func="changes_locals_twice") as t:
        changes_locals_twice()
    assert t.getLVVmap() == {"a": 2, "b": 2, "c": 2}


def test_does_not_track_changes_to_equal():
    # Assumption 3: local variables which change to something == don't count as a change
    with LVVTracer(target_func="changes_to_equal") as t:
        changes_to_equal()
    assert t.getLVVmap() == {"a": 1}


def test_tracks_only_target_function():
    # Assumption 4: only changes in the traced_function are tracked
    with LVVTracer(target_func="changes_locals_twice") as t:
        changes_locals_once()
        changes_locals_twice()
        changes_to_equal()
    assert t.getLVVmap() == {"a": 2, "b": 2, "c": 2}


def test_does_not_track_callees_of_target_function():
    # Assumption 5: calls in the target function are ignored
    with LVVTracer(target_func="calls_other_function") as t:
        calls_other_function()
    assert t.getLVVmap() == {"a": 1}


def test_tracks_changes_in_mutated_objects():
    # Assumption 6: changes are tracked if `is` and `!=`
    with LVVTracer(target_func="changes_mutable") as t:
        changes_mutable()
    assert t.getLVVmap() == {"a": 3}


def test_does_not_track_identity_changes():
    # Assumption 7: changes are not tracked if `==` and `is not`
    with LVVTracer(target_func="changes_identity_but_not_equality") as t:
        changes_identity_but_not_equality()
    assert t.getLVVmap() == {"a": 1}


def test_tracks_multiple_calls_sums_changes_in_all_calls():
    # Assumption 8: calling the same function multiple times returns the sum of changes across all calls
    with LVVTracer(target_func="changes_locals_once") as t:
        changes_locals_once()
        changes_locals_once()
        changes_locals_once()
    assert t.getLVVmap() == {"a": 3, "b": 3, "c": 3}

    with LVVTracer(target_func="changes_locals_twice") as t:
        changes_locals_twice()
        changes_locals_twice()
    assert t.getLVVmap() == {"a": 4, "b": 4, "c": 4}

    with LVVTracer(target_func="changes_a_but_not_b") as t:
        changes_a_but_not_b()
        changes_a_but_not_b()
    assert t.getLVVmap() == {"a": 4, "b": 2}


def test_treats_nonlocal_statement_as_definition():
    # Assumption 9: `nonlocal` statements are treated as if a local variable was defined
    a = 1
    def changes_non_local():
        nonlocal a
        a = 5

    with LVVTracer(target_func="changes_non_local") as t:
        changes_non_local()
    assert t.getLVVmap() == {"a": 2}


def test_ignores_globals():
    # Assumption 10: global variables are ignored
    global_var = 1

    with LVVTracer(target_func="changes_global") as t:
        changes_global()
    assert t.getLVVmap() == {}


def test_argument_as_definition():
    # Assumption 11: arguments are treated as locals; counting their definition on function entry
    with LVVTracer(target_func="does_not_change_argument") as t:
        does_not_change_argument(1)
    assert t.getLVVmap() == {"arg": 1}

    # Assumption 11a: even if default arguments are provided
    with LVVTracer(target_func="does_not_change_argument") as t:
        does_not_change_argument()
    assert t.getLVVmap() == {"arg": 1}


def test_changes_argument():
    # Assumption 12: arguments are treated as locals
    with LVVTracer(target_func="changes_argument") as t:
        changes_argument(1)
    assert t.getLVVmap() == {"arg": 2}

    # Assumption 12a: even if default arguments are provided
    with LVVTracer(target_func="changes_argument") as t:
        changes_argument()
    assert t.getLVVmap() == {"arg": 2}


def test_treats_recursive_calls_to_itself():
    # Assumption 13: Recursive calls are tracked
    with LVVTracer(target_func="calls_itself_recursively_six_times") as t:
        calls_itself_recursively_six_times()

    # NOTE: Returning from the recursive call brings back the upper value of `n` - hence `*2`,
    #       except for the base call - hence - 1.
    assert t.getLVVmap() == {"n": 6*2 - 1}


def test_deepcopy_quirk():
    # Assumption 14: objects without __eq__ change with every traced line
    with LVVTracer(target_func="does_not_change_identity_and_equality") as t:
        does_not_change_identity_and_equality()
    assert t.getLVVmap() == {"a": 3}
