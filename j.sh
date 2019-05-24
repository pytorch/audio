# find all the test files that match "test*.py"
TEST_FILES="$(find test/ -type f -name "test*.py" | sort)"
echo "Test files are:"
echo $TEST_FILES

echo "Executing tests:"
EXIT_STATUS=0
for FILE in $TEST_FILES; do
  # run each file on a separate process. if one fails, just keep going and
  # return the final exit status.
  python -m unittest -v $FILE
  STATUS=$?
  EXIT_STATUS="$(($EXIT_STATUS+STATUS))"
done

exit $EXIT_STATUS
