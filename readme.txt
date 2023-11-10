This txt-file contains explanations of how to deal with the current project

###########################################################################
Congrats: if you can see this file you've already cloned the repository and will be able to use the current project very soon.
To get the full experience:
	1. Open setup.bat with a text editor and set the path where the python compiler you want to use is located on your machine.
	2. Save setup.bat
	3. Doubleclick on setup.bat



###########################################################################
How to create a similar project:

	1. Add to your folder the files setup.bat, .gitignore and readme.txt
	2. Open folder in VSCode
	3. Click on SourceControl-Button in VSCode and click on Initialize Repository
	(4. Choose the folder from workspace) 
	5. Commit with hook symbol including comment
	6. Click on [View] in VSCode -> Command Pallete -> Add: Remote Repository
	7. Create Github Repository and connect
	8. Push




###########################################################################
Work with git:
$ git init							//Initialize Local Git Repository
$ git add <file> 						//Add File(s) to Index
$ git status							//Check Status of Working Tree
$ git commit							//Commit Changes In Index -> press "i" for Insert-mode -> Edit commit -> [ESC] finishes insert mode -> :wq executes the commit 
$ git commit -m 'Description' 					//Easy commit with description
$ git push							//Push to remote Repository
$ git pull							//Pull latest from remote repository
$ git clone							//Clone Repository into a new directory
$ git config --global user.name 'Stephan'			//Fügt Benutzernamen hinzu
$ git config --global user.email 'stephan.hoetger@kabelmail.de'	//Fügt E-Mail-Adresse hinzu
$ git rm --cached
$ git add *.ending						//Will add any file with certain ending
$ touch .gitignore						//Creates .gitignore file 
$ git branch side_branch					//Creates side branch
$ git checkout side_branch					//Changes to side branch
$ git merge side_branch						//Merges side_branch to master branch
$ git remote							// Show remote repositories

#Connect repository with github account
	1. Got to github.com
	2. Create new repository
	3. Follow instructions
		3.1 git remote add origin https://github.com/...
		3.2 git push -u origin main