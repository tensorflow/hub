<!--* freshness: { owner: 'maringeo' reviewed: '2021-05-28' review_interval: '6 months' } *-->

# Contribute a model

This page is about adding Markdown documentation files to GitHub. For more
information on how to write the Markdown files in the first place, please see
the [writing model documentation guide](writing_model_documentation.md).

## Submitting the model

The complete Markdown files can be pulled into the master branch of
[tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev/tree/master) by
one of the following methods.

### Git CLI submission

Assuming the identified markdown file path is
`assets/docs/publisher/model/1.md`, you can follow the standard Git[Hub]
steps to create a new Pull Request with a newly added file.

This starts with forking the TensorFlow Hub GitHub repository, then creating a
[Pull Request from this fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)
into the TensorFlow Hub master branch.

The following are typical CLI git commands needed to adding a new file to a
master branch of the forked repository.

```bash
git clone https://github.com/[github_username]/tfhub.dev.git
cd tfhub.dev
mkdir -p assets/docs/publisher/model
cp my_markdown_file.md ./assets/docs/publisher/model/1.md
git add *
git commit -m "Added model file."
git push origin master
```

### GitHub GUI submission

A somewhat more straightforward way of submitting is via GitHub graphical user
interface. GitHub allows creating PRs for
[new files](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files)
or
[file edits](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository)
directly through GUI.

1.  On the [TensorFlow Hub GitHub page](https://github.com/tensorflow/tfhub.dev),
    press `Create new file` button.
1.  Set the right file path: `assets/docs/publisher/model/1.md`
1.  Copy-paste the existing markdown.
1.  At the bottom, select "Create a new branch for this commit and start a pull
    request."
