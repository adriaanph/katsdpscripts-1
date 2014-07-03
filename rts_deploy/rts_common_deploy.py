from fabric.api import sudo, run, env
from fabric.contrib import files
import os

#GIT branch to use for deployment
GIT_BRANCH = 'master'

# oodt install info
OODT_HOME = '/usr/local/oodt/'
STAGING_HOME = '/var/kat/data'
ARCHIVE_HOME = '/var/kat/archive'
SDP_MC = '/var/kat/sdpcontroller'
ARCHIVE_APACHE_HOME = '/var/www/archive'
# Subdirectories derived from roots
# Where we're going to put the oodt configuration files
OODT_CONF = os.path.join(OODT_HOME, "conf")
# Primary staging directory
SOLR_COLLECTIONS_HOME = os.path.join(ARCHIVE_HOME, "catalogs/solr")
STAGING_INGEST = os.path.join(STAGING_HOME, "staging")
# Where files go when ingest fails
STAGING_FAILED = os.path.join(STAGING_HOME, "failed")
# Secondary staging directory for remote archiving
STAGING_NFS_INGEST = os.path.join(STAGING_HOME, "nfs_staging")
# Archive directory where all the data products will be stored
ARCHIVE_DATA = os.path.join(ARCHIVE_HOME, "data")
# RTS storage directory where all the data products will be stored
RTS_DATA = os.path.join(ARCHIVE_DATA, "RTS")

#Solr and Tomcat
SOLR_VER = "4.4.0"
TOMCAT_VER = "7.0.42"
TOMCAT_CONF = os.path.join(OODT_HOME, 'apache-tomcat/conf/server.xml')

def install_deb_packages(packages):
    """Install debian packages listed in space-separated string"""
    print ' ---- Install debian packages ---- \n', packages, '\n'
    sudo('yes | DEBIAN_FRONTEND=noninteractive apt-get install %s' % (packages))

def remove_deb_packages(packages):
    """Remove debian packages listed in space-separated string"""
    print ' ---- Remove debian packages ---- \n', packages, '\n'
    sudo('yes | DEBIAN_FRONTEND=noninteractive apt-get remove %s' % (packages))

def install_pip_packages(packages):
    """Pip install packages listed in space-separated string"""
    print ' ---- Install', packages, ' ---- \n'
    sudo('pip install -U %s' % (packages,))

def remove_pip_packages(packages):
    """Pip uninstall packages listed in space-separated string"""
    print ' ---- Uninstall', packages, ' ---- \n'
    # check if the deploy name is different to the package name
    try:
        sudo('yes | pip uninstall %s' % (packages))
    except:
        # we don't get a nice named exception if the package isn't there
        print 'Cannot uninstall \n'

def install_git_package(package, repo='ska-sa', user='katpull', password='katpull4git', branch='master',**kwargs):
    install_pip_packages('git+https://'+user+':'+password+'@github.com/'+repo+'/'+package+'.git'+'@'+branch)

def remove_dir(rmdir):
    sudo("rm -rf %s" % (rmdir,))

def make_directory(dir_to_make):
    sudo("mkdir -p %s" % (dir_to_make))
    sudo("chown -R %s %s" % (env.user, dir_to_make))
    sudo("chgrp -R %s %s" % (env.user, dir_to_make))

def make_dc_directory_trees():
    #staging directories
    make_directory(STAGING_INGEST)
    make_directory(STAGING_NFS_INGEST)
    make_directory(STAGING_FAILED)
    make_directory(ARCHIVE_DATA)
    make_directory(SOLR_COLLECTIONS_HOME)
    make_directory(RTS_DATA)
    make_directory(SDP_MC)

def configure_tomcat():
    files.sed(TOMCAT_CONF,
              '<Connector port="8080" protocol="HTTP/1.1"',
              '<Connector port="8983" protocol="HTTP/1.1"',
              use_sudo=True,
              backup='.bak')

def deploy_oodt_comp(comp_to_install, comp_ver):
    comp_tar = "%s-dist.tar.gz" % (comp_ver)
    run("rm -rf %s" % (os.path.join(OODT_HOME, comp_to_install)))
    run("wget -O /tmp/%s http://kat-archive.kat.ac.za/oodt_installs/%s" % (comp_tar, comp_tar))
    run("tar xzvf /tmp/%s -C %s" % (comp_tar, OODT_HOME))
    run("mv %s %s" % (os.path.join(OODT_HOME, comp_ver), os.path.join(OODT_HOME, comp_to_install)))
    run("rm -f /tmp/%s" % (comp_tar))

def deploy_oodt_comp_ver_06(comp_to_install):
    deploy_oodt_comp(comp_to_install, "%s-0.6" % (comp_to_install))

def deploy_dc_oodt_comps():
    deploy_oodt_comp_ver_06("cas-filemgr")
    deploy_oodt_comp_ver_06("cas-crawler")
    deploy_tomcat()
    deploy_solr()

def deploy_oodt_conf(branch):
    """Deploy the oodt configuration for branch (i.e karoo/vkaroo/lab/comm VMs).

    Parameters
    ==========
    branch: The OODT configuration svn branch to checkout. OODT configuration can be
    found in '../svnDS/code/katconfig/{branch}/static/oodt/'. For example 'trunk' or
    'branches/karoo'.
    """
    run("rm -rf %s" % (OODT_CONF))
    run("svn co --username kat --password kat "
        "https://katfs.kat.ac.za/svnDS/code/katconfig/%s/static/oodt/ "
        "%s" % (branch, OODT_CONF))

def deploy_tomcat(comp_to_install="apache-tomcat"):
    deploy_oodt_comp(comp_to_install, "%s-%s" % (comp_to_install, TOMCAT_VER))

def deploy_solr(comp_to_install="solr"):
    deploy_oodt_comp(comp_to_install, "%s-%s" % (comp_to_install, SOLR_VER))

def deploy_solr_in_tomcat():
    run("cp %s/solr/dist/solr-%s.war %s/apache-tomcat/webapps/solr.war" % (OODT_HOME, SOLR_VER, OODT_HOME))
    run("cp %s/solr/example/lib/ext/* %s/apache-tomcat/lib" % (OODT_HOME, OODT_HOME))
    run("rsync -rv --progress %s/solr/ %s" % (OODT_CONF, SOLR_COLLECTIONS_HOME))
    run("rm -rf %s/solr " % (OODT_HOME))

def check_and_make_sym_link(L_src, L_dest):
    sudo("if [[ ! -L %s ]]; then ln -s %s %s; fi" % (L_dest, L_src, L_dest))

def export_dc_nfs_staging():
    files.append('/etc/exports',
                 '%s rts-imager.kat.ac.za(rw,sync,no_subtree_check)' % (STAGING_NFS_INGEST),
                 use_sudo=True)
    sudo('exportfs -a')

def oodt_setup():
    make_directory(OODT_HOME)
    make_dc_directory_trees()
    deploy_dc_oodt_comps()
    deploy_oodt_conf('branches/oodt_0_6_upgrade')
    configure_tomcat()
    deploy_solr_in_tomcat()
    export_dc_nfs_staging()

def remove_oodt_directories():
    remove_dir(OODT_HOME)
    remove_dir(ARCHIVE_HOME)
    remove_dir(ARCHIVE_APACHE_HOME)
    remove_dir(STAGING_HOME)
    remove_dir(SDP_MC)

