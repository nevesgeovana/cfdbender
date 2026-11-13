Applied storage and record management operations hold the destination manifest's
writer lock from selection through file changes and record replacement. Nested
registry writes retain the outer lock. A competing writer refuses before changes.
