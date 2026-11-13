SU2 and OpenFOAM warn and skip unread reference and post keys instead of refusing
them. Mock now warns about its unread keys too. The C-09 test covers both files
on all six adapters and verifies that translation still succeeds.
