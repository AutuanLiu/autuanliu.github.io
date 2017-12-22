# Docker
## Docker 基础

* 国内源

```
curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg | sudo apt-key add
-
```

```
sudo add-apt-repository \
"deb [arch=amd64] https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu \
$(lsb_release -cs) \
stable"
```

```
sudo apt update
```

```
sudo apt install docker-ce
```

1. 使用脚本自动安装

```
$ curl -fsSL get.docker.com -o get-docker.sh
$ sudo sh get-docker.sh --mirror Aliyun
```

2. 启动

```shell
$ sudo systemctl enable docker
$ sudo systemctl start docker
```

3. 建立 docker 用户组

出于安全考虑，一般 Linux 系统上不会直接使用 root 用户。因此，更好地做法是将需要使用 docker的用户加入 docker用户组

```
$ sudo groupadd docker
```

4. 将当前用户加入 docker 组

```
sudo usermod -aG docker $USER
```

5. 设置国内镜像

[Docker 镜像加速器-博客-云栖社区-阿里云](https://yq.aliyun.com/articles/29941?spm=5176.100239.blogcont7697.20.WlzdBn)

* 您可以添加"https://registry.docker-cn.com"到registry-mirrors数组中/etc/docker/daemon.json 以默认从China注册表镜像中拉取。

    ```
    {
     "registry-mirrors": [
        "https://registry.docker-cn.com"
        ]
    }
    ```

[Registry as a pull through cache | Docker Documentation](https://docs.docker.com/registry/recipes/mirror/#use-case-the-china-registry-mirror)

* 之后重新启动服务

```
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

### 获取镜像


```
dcoker pull [选项] [Docker Registy 地址]<仓库名>:<tag>
```

Docker Registy 地址]格式为：IP/域名[:端口号] 默认是 docker hub

仓库名：用户名/软件名


镜像是有**多层存储** 所构成，下载也是一层一层下载的，并非单一的文件，下载的过程中会给出每一层的ID的前12位


有了镜像后就可以**以镜像为基础**创建一个**容器**来运行


### 使用 Docker image 来管理镜像

```
docker image pull ubuntu:17.10 拉取镜像
docker container run -it --rm ubuntu:17.10 bash   运行容器
```

-it 交互式的终端

--rm 运行完 删除

bash 容器内命令


* 列出镜像

`docker images`

一个镜像可以拥有多个 tag 如 ubuntu:17.10 和 latest 是相同的


`docker system df` 查看 镜像，容器 所占体积


* 虚悬镜像：没有仓库名，名优标签，均为 none ，docker pull 和 docker build 均会导致这种情况
* 因为新旧镜像同名，导致旧镜像成为虚悬镜像
* docker images -f dangling=true 专门显示这类镜像
* docker image prunge 删除虚悬镜像


* 为了加速镜像的构建，Docker会利用中间层镜像，默认只显示最顶层镜像
* dokcer images -a 显示中间层
* 这里没有标签的镜像是不能删除的


2. 列出部分镜像

    1. `docker images ubuntu `
    2. `docker images ubuntu:16.04`
    3. `docker images -f` 过滤器
    4. `docker images -f since=-mongo:3.2` mongo之后建立的镜像(before)
    5. `docker images -q` 只显示 id
    6. `docker images --format "{{.ID}:{.Repository}}"` (GO的模板语法)

3. Docker 1.13 +

`dokcer image ls`

4. 镜像是多层存储，每一层在其前一层基础上进行修改，容器同样是 多层存储，以镜像为基础层，在其基础上再加一层作为容器运行的存储层

5. 修改了容器的文件也就是改动了容器的存储层 `docker diff` 查看改动
6. --name xx 指定容器的名字为 xx （）**最好指定名字便于操作**）
7. `docker exec -it containerName bash` 进入容器并执行 bash
6. `docker commit` 可以降容器的**存储层保存下来成为镜像，以后运行这个新镜像的时候，就会拥有原有镜像的文件变化** 
    1. docker commit [选项] <容器ID或者容器名> [<仓库名>[:<tag>]]
    2. 将容器保存为镜像
    ```
    docker commit \
    --author "autuanliu <autuanliu@163.com>" \
    --message "fix " \ 可以不写
    webserver \  容器名字
    nginx:v2 指定仓库名:tag
    ```
8. **这里一定要记得写一个++保存脚本++啊！！！很容易忘记啊，直接就关闭了，然后什么也没有了啊**
9. `Docker history 仓库名:tag` 查看历史 


* **一定要慎用 commit 虽然 commit 可以添加内容，构建新的镜像，但是同时也会添加一些无关紧要的东西，是的镜像变得臃肿**
* `docker commit` 意味着对镜像的操作都是一个黑箱操作，生成的镜像也被成为黑箱镜像，也就是说：除了构建镜像的人知道执行了什么操作，其他人根本无从得知

## 操作 Docker 容器

容器是独立运行的一个或一组应用，以及它们的运行态环境

### 启动容器

启动容器有两种方式，一种是基于镜像新建一个容器并启动，另外一个是将在终止状态（stopped） 的容器重新启动

```docker run -t -i ubuntu:14.04 /bin/bash```

其中， -t 选项让Docker分配一个伪终端（pseudo-tty） 并绑定到容器的标准输入上， -i 则让容器的标准输入保持打开。在交互模式下，用户可以通过所创建的终端来输入命令

* 检查本地是否存在指定的镜像，不存在就从公有仓库下载
* 利用镜像创建并启动一个容器
* 分配一个文件系统，并在只读的镜像层外面挂载一层可读写层
* 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去
* 从地址池配置一个 ip 地址给容器
* 执行用户指定的应用程序
* 执行完毕后容器被终止

### 启动已终止容器

可以利用 docker start 命令，直接将一个已经终止的容器启动运行

### 后台(background)运行

更多的时候，需要让 Docker 在后台运行而不是直接把执行命令的结果输出在当前宿主机下。此时，可以通过添加 -d 参数来实现

* 要获取容器的输出信息，可以通过 docker logs 命令
    
    ```docker logs [container ID or NAMES]```

### 终止容器

* 可以使用 docker stop 来终止一个运行中的容器。此外，当Docker容器中指定的应用终结时，容器也自动终止。 例如对于上一章节中只启动了一个终端的容器，用户通过 exit 命令或 Ctrl+d 来退出终端时，所创建的容器立刻终止。
* 终止状态的容器可以用 docker ps -a 命令看到
* 处于终止状态的容器，可以通过 docker start 命令来重新启动。
* docker restart 命令会将一个运行态的容器终止，然后再重新启动它

### 进入容器

* docker attach name
* nsenter 启动一个新的shell进程(默认是/bin/bash), 同时会把这个新进程切换到和目标(target)进程相同的命名空间，这样就相当于进入了容器内部。nsenter 要正常工作需要有 root 权限

### 导出容器

如果要导出本地某个容器，可以使用 docker export 命令

```docker export cintainerID_name > ubuntu.tar```

### 导入容器快照

可以使用 docker import 从容器快照文件中再导入为镜像，例如
```$ cat ubuntu.tar | docker import - test/ubuntu:v1.0```

* 也可以通过指定 URL 或者某个目录来导入，例如

```$ docker import http://example.com/exampleimage.tgz example/imagerepo```
```
$ docker container export
$ docker image import
```

* `docker container prune` 可以清理掉所有处于终止状态的容器


### 仓库

一个容易混淆的概念是注册服务器（Registry） 。实际上注册服务器是管理仓库的具体服务器，每个服务器上可以有多个仓库，而每个仓库下面有多个镜像。从这方面来说，仓库可以被认为是一个具体的项目或目录。例如对于仓库地址 dl.dockerpool.com/ubuntu 来说， dl.dockerpool.com 是注册服务器地址， ubuntu 是仓库名

### 登录

* 可以通过执行 docker login 命令交互式的输入用户名及密码来完成在命令行界面的登录。登录成功后，本地用户目录的 .dockercfg 中将保存用户的认证信息
* docker search 命令来查找官方仓库中的镜像，并利用 docker pull 命令来将它下载到本地

### Docker 数据管理

在容器中管理数据主要有两种方式：
* 数据卷（Volumes）
* 挂载主机目录 (Bind mounts)
* 使用 docker volume 子命令来管理 Docker 数据卷

数据卷是一个可供一个或多个容器使用的特殊目录，它绕过 UFS，可以提供很多有用的特性：
* 数据卷可以在容器之间共享和重用
* 对数据卷的修改会立马生效
* 对数据卷的更新，不会影响镜像
* 数据卷默认会一直存在，即使容器被删除
* 数据卷的使用，类似于 Linux 下对目录或文件进行 mount，镜像中的被指定为挂载点的目录中的文件会被隐藏掉，挂载的数据卷的内容会被映射到该目录
* **Docker 新用户应该选择 --mount 参数，经验丰富的 Dcoker 使用者对 -v 或者 --volume已经很熟悉了，但是推荐使用 --volume 参数**

### 创建一个数据卷

```$ docker volume create my-vol```

* 查看所有的数据卷

```$ docker volume ls```

* 查看指定数据卷的信息

```docker volume inspect my-vol```

### 创建一个指定位置的数据卷

```
docker volume create --name sharef --opt type=none --opt device=~/home/autuanliu/sharef --opt o=bind

```


### 启动一个挂载数据卷的容器

* 在用 docker run 命令的时候，使用 --mount 标记来将数据卷挂载到容器里。在一次 docker run 中可以挂载多个数据卷
    ```
    docker run -d -P \
    --name web \ 容器名
    --mount source=my-vol,target=/webapp \  加载一个数据卷到容器的 /webapp 目录
    training/webapp \  镜像名
    python app.py \ 在 bash 执行命令
    ```
* 查看数据卷的具体信息

    ```$ docker inspect web```
### 删除数据卷

```$ docker volume rm my-vol```

数据卷是被设计用来**持久化数据**的，++它的生命周期独立于容器++，Docker 不会在容器被删除后自动删除数据卷，并且也不存在垃圾回收这样的机制来处理没有任何容器引用的数据卷。

如果需要在删除容器的同时移除数据卷。可以在删除容器的时候使用` docker rm -v `这个命令

* 无主的数据卷可能会占据很多空间，要清理请使用以下命令

    ```docker volume prune```

### 挂载一个主机目录作为数据卷

* 使用 --mount 标记可以指定挂载一个本地主机的目录到容器中去。

```
$ docker run -d -P \
--name web \
# -v /src/webapp:/opt/webapp \
--mount type=bind,source=/src/webapp,target=/opt/webapp \ 加载主机的 /src/webapp 目录到容器的 /opt/webapp 目录,本
地目录的路径必须是绝对路径，如果目录不存在 Docker 会自动为你创建它
training/webapp \
python app.py
```

Docker 挂载主机目录的默认权限是读写，用户也可以通过增加 readonly 指定为只读

```--mount type=bind,source=/src/webapp,target=/opt/webapp,readonly ```

### 挂载一个本地主机文件作为数据卷

--mount 标记也可以从主机挂载单个文件到容器中

```
$ docker run --rm -it \
--mount type=bind,source=~/.bash_history,target=/root/.bash_history \
ubuntu:17.10 \
bash
```

这样就可以记录在容器输入过的命令了

### 外部访问容器

容器中可以运行一些网络应用，要让外部也可以访问这些应用，可以通过 -P 或 -p 参数来指定端口映射

当使用 -P 标记时，Docker 会随机映射一个 49000~49900 的端口到内部容器开放的网络端口

可以通过 docker logs 命令来查看应用的信息

-p（小写的） 则可以指定要映射的端口，并且，在一个指定端口上只可以绑定一个容器。支持
的格式有 ip:hostPort:containerPort | ip::containerPort | hostPort:containerPort

* 映射所有接口地址
    * 使用 hostPort:containerPort 格式本地的 5000 端口映射到容器的 5000 端口，可以执行` docker run -d -p 5000:5000 training/webapp python app.py`此时默认会绑定本地所有接口上的所有地址
* 映射到指定地址的指定端口
    * 可以使用 ip:hostPort:containerPort 格式指定映射使用一个特定地址，比如 localhost 地址 127.0.0.1
* 映射到指定地址的任意端口
    * 使用 ip::containerPort 绑定 localhost 的任意端口到容器的 5000 端口，本地主机会自动分配一个端口。
* 还可以使用 udp 标记来指定 udp 端口    
    * `$ docker run -d -p 127.0.0.1:5000:5000/udp training/webapp python app.py` 
* 查看映射端口配置
    * 使用 docker port 来查看当前映射的端口配置，也可以查看到绑定的地址 ` docker port nostalgic_morse 5000`
* 容器有自己的内部网络和 ip 地址（使用 docker inspect 可以获取所有的变量，Docker还可以有一个可变的网络配置）
* -p 标记可以多次使用来绑定多个端口

### 容器互联

容器的连接（linking） 系统是除了端口映射外，另一种跟容器中应用交互的方式。该系统会在源和接收容器之间创建一个隧道，接收容器可以看到源容器指定的信息。
自定义容器命名

连接系统依据容器的名称来执行。因此，首先需要自定义一个好记的容器命名。

* 虽然当创建容器的时候，系统默认会分配一个名字。自定义命名容器有2个好处：
    * 自定义的命名，比较好记，比如一个web应用容器我们可以给它起名叫web
    * 当要连接其他容器时候，可以作为一个有用的参考点，比如连接web容器到db容器
    * 使用 --name 标记可以为容器自定义命名。
     ```$ docker run -d -P --name web training/webapp python app.py```
* 在执行 docker run 的时候如果添加 --rm 标记，则容器在终止后会立刻删除
* --rm 和 -d 参数不能同时使用

### 容器互联

* 使用 --link 参数可以让容器之间安全的进行交互

创建一个新的 web 容器，并将它连接到 db 容器

```$ docker run -d -P --name web --link db:db training/webapp python app.py```

--link 参数的格式为 --link name:alias ，其中 name 是要链接的容器的名称， alias 是这个连接的别名

Docker 在两个互联的容器之间创建了一个安全隧道，而且不用映射它们的端口到宿主主机上。在启动 db 容器的时候并没有使用 -p 和 -P 标记，从而避免了暴露数据库端口到外部网络上

* Docker 通过 2 种方式为容器公开连接信息：
    * 环境变量
    * 更新 /etc/hosts 文件

使用 env 命令来查看 web 容器的环境变量 

`docker run --rm --name web2 --link db:db training/webapp env`

除了环境变量，Docker 还添加 host 信息到父容器的 /etc/hosts 的文件

### 容器访问外部网络

容器要想访问外部网络，需要本地系统的转发支持。在Linux 系统中，检查转发是否打开。

```
$sysctl net.ipv4.ip_forward
net.ipv4.ip_forward = 1
```

如果为 0，说明没有开启转发，则需要手动打开。

```$sysctl -w net.ipv4.ip_forward=1```

如果在启动 Docker 服务的时候设定 --ip-forward=true , Docker 就会自动设定系统的ip_forward 参数为 1

## 使用 dockerfile 定制镜像

镜像的定制实际上就是定制每一层所添加的配置、文件，如果我们可以把每一层修改、安装、构建、操作的命令都写入一个脚本，用这个脚本来构建、定制镜像，那么之前提及的无法重复的问题、镜像构建透明性的问题、体积的问题就都会解决。这个脚本就是 Dockerfile

Dockerfile 是一个文本文件，其内包含了一条条的指令(Instruction)，每一条指令构建一层，因此每一条指令的内容，就是描述该层应当如何构建

### FROM

FROM 就是指定基础镜像，因此一个 Dockerfile 中 FROM 是必备的指令，并且必须是第一条指令

除了选择现有镜像为基础镜像外，Docker 还存在一个特殊的镜像，名为 scratch 。这个镜像是虚拟的概念，并不实际存在，它表示一个空白的镜像，如果你以 scratch 为基础镜像的话，意味着你不以任何镜像为基础，接下来所写的指令将作为镜像第一层开始存在

对于 Linux 下静态编译的程序来说，并不需要有操作系统提供运行时支持，所需的一切库都已经在可执行文件里了，因此直接 FROM scratch 会让镜像体积更加小巧。使用 Go 语言 开发的应用很多会使用这种方式来制作镜像，这也是为什么有人认为 Go 是特别适合容器微服务架构的语言的原因之一

### RUN

RUN 指令是用来执行命令行命令的。由于命令行的强大能力， RUN 指令在定制镜像时是最常用的指令之一

* 格式
    * shell 格式： RUN <命令> ，就像直接在命令行中输入的命令一样
    
    ```RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html```

    * exec 格式： RUN ["可执行文件", "参数1", "参数2"] ，这更像是函数调用中的格式
    
    * **Dockerfile 中每一个指令都会建立一层， RUN 也不例外**

使用一个 RUN 指令，并使用 && 将各个所需命令串联起来。将之前的 7 层，简化为了1 层。在撰写 Dockerfile 的时候，要经常提醒自己，这并不是在写 Shell 脚本，而是在定义每一层该如何构建。换行符 \

Dockerfile 支持 Shell 类的行尾添加 \ 的命令换行方式，以及行首 # 进行注释的格式。良好的格式，比如换行、缩进、注释等，会让维护、排障更为容易，这是一个比较好的习惯。

```shell
FROM debian:jessie
RUN buildDeps='gcc libc6-dev make' \
&& apt-get update \
&& apt-get install -y $buildDeps \
&& wget -O redis.tar.gz "http://download.redis
&& mkdir -p /usr/src/redis \
&& tar -xzf redis.tar.gz -C /usr/src/redis --s
&& make -C /usr/src/redis \
&& make -C /usr/src/redis install \
&& rm -rf /var/lib/apt/lists/* \
&& rm redis.tar.gz \
&& rm -r /usr/src/redis \
&& apt-get purge -y --auto-remove $buildDeps
```

这一组命令的最后添加了清理工作的命令，删除了为了编译构建所需要的软件，清理了所有下载、展开的文件，并且还清理了 apt 缓存文件。这是很重要的一步，我
们之前说过，**镜像是多层存储，每一层的东西并不会在下一层被删除，会一直跟随着镜像。**
++==因此镜像构建时，一定要确保每一层只添加真正需要添加的东西，任何无关的东西都应该清理掉==++

### 构建镜像

在 Dockerfile 文件所在目录执行：`docker build -t 仓库名:tag`

`docker build [选项] <上下文路径/URL/->`

### 镜像构建上下文（Context）

如果注意，会看到 docker build 命令最后有一个 . , . 表示当前目录，而 Dockerfile 就在当前目录，因此不少初学者以为这个路径是在指定 Dockerfile 所在路径，这么理解其实是不准确的。如果对应上面的命令格式，你可能会发现，这是在指定上下文路径

Docker 在运行时分为 Docker 引擎（也就是服务端守护进程） 和客户端工具。Docker 的引擎提供了一组 REST API，被称为 Docker
Remote API，而如 docker 命令这样的客户端工具，则是通过这组 API 与 Docker 引擎交互，从而完成各种功能。因此，虽然表面上我们好像是在本机执行各种 docker 功能，但实际上，一切都是使用的远程调用形式在服务端（Docker 引擎） 完成。也因为这种 C/S 设计，让我们操作远程服务器的 Docker 引擎变得轻而易举

当我们进行镜像构建的时候，并非所有定制都会通过 RUN 指令完成，经常会需要将一些本地文件复制进镜像，比如通过 COPY 指令、 ADD 指令等。而 docker build 命令构建镜像，其实并非在本地构建，而是在服务端，也就是 Docker 引擎中构建的

当构建的时候，用户会指定构建镜像上下文的路径， docker build 命令得知这个路径后，会将路径下的所有内容打包，然后上传给 Docker 引擎。这样
Docker 引擎收到这个上下文包后，就会获得构建镜像所需的一切文件。

`COPY ./package.json /app/` 这并不是要复制执行 docker build 命令所在的目录下的 package.json ，也不是复制 Dockerfile 所在目录下的 package.json ，而是复制 上下文（context） 目录下的 package.json

COPY 这类指令中的源文件的路径都是相对路径。COPY ../package.json /app 或者 COPY /opt/xxxx /app 无法工作的是因为这些路径已经
超出了上下文的范围，Docker 引擎无法获得这些位置的文件。如果真的需要那些文件，应该将它们复制到上下文目录中去

一般来说，应该会将 Dockerfile 置于一个空目录下，或者项目根目录下。如果该目录下没有所需文件，那么应该把所需文件复制一份过来。如果目录下有些东西确实不希望构建时传给 Docker 引擎，那么**可以用 .gitignore 一样的语法写一个 .dockerignore ，该文件是用于剔除不需要作为上下文传递给 Docker 引擎的**

Dockerfile 的文件名并不要求必须为 Dockerfile ，而且并不要求必须位于上下文目录中，比如可以用 -f ../Dockerfile.php 参数指定某个文件作为Dockerfile 

* 直接用 Git repo 进行构建

`$ docker build https://github.com/twang2218/gitlab-ce-zh.git#:8.14`

这行命令指定了构建所需的 Git repo，并且指定默认的 master 分支，构建目录为 /8.14/ ，然后 Docker 就会自己去 git clone 这个项目、切换到指定分支、并进入到指定目录后开始构建

* 用给定的 tar 压缩包构建

`$ docker build http://server/context.tar.gz`

如果所给出的 URL 不是个 Git repo，而是个 tar 压缩包，那么 Docker 引擎会下载这个包，并自动解压缩，以其作为上下文，开始构建

* 从标准输入中读取 Dockerfile 进行构建

`docker build - < Dockerfile`
或
`cat Dockerfile | docker build -`

**如果标准输入传入的是文本文件，则将其视为 Dockerfile ，并开始构建**。这种形式由于直接从标准输入中读取 Dockerfile 的内容，**它没有上下文，因此不可以像其他方法那样可以将本地文件 COPY 进镜像之类的事情**

* 从标准输入中读取上下文压缩包进行构建

`$ docker build - < context.tar.gz`

如果发现标准输入的文件格式是 gzip 、 bzip2 以及 xz 的话，将会使其为上下文压缩包，直接将其展开，将里面视为上下文，并开始构建

**目前的主流方法 一般使用 docker + Management Command + Command 来进行操作**

### COPY 复制文件

格式：

COPY <源路径>... <目标路径>

COPY ["<源路径1>",... "<目标路径>"]

COPY 指令将从构建上下文目录中 <源路径> 的文件/目录复制到新的一层的镜像内的 <目标路径> 位置
```
COPY package.json /usr/src/app/
```
<源路径> 可以是多个，甚至可以是通配符，其通配符规则要满足 Go 的 filepath.Match 规则

使用 COPY 指令，源文件的各种元数据都会保留。比如读、写、执行权限、文件变更时间等

### ADD 更高级的复制文件

* <源路径> 可以是一个 URL ，这种情况下，Docker 引擎会试图去下载这个链接的文件放到 <目标路径> 去
* 下载后的文件权限自动设置为 600 ，如果这并不是想要的权限，那么还需要增加额外的一层 RUN 进行权限调整
* 如果下载的是个压缩包，需要解压缩，也一样还需要额外的一层 RUN 指令进行解压缩
* 如果 <源路径> 为一个 tar 压缩文件的话，压缩格式为 gzip , bzip2 以及 xz 的情况下， ADD 指令将会自动解压缩这个压缩文件到 <目标路径> 去
* 这个功能其实并不实用，而且不推荐使用

### CMD 容器启动命令

```
CMD 指令的格式和 RUN 相似，也是两种格式：
shell 格式： CMD <命令>
exec 格式： CMD ["可执行文件", "参数1", "参数2"...]
参数列表格式： CMD ["参数1", "参数2"...] 。在指定了 ENTRYPOINT 指令后，用 CMD 指
定具体的参数
```

* ubuntu 镜像默认的 CMD 是 /bin/bash 
* Docker 不是虚拟机，容器中的应用都应该以前台执行，而不是像虚拟机、物理机里面那样，用 upstart/systemd 去启动后台服务，容器内没有后台服务的概念
* 对于容器而言，其启动程序就是容器应用进程，容器就是为了主进程而存在的，主进程退出，容器就失去了存在的意义，从而退出，其它辅助进程不是它需要关心的东西

### ENTRYPOINT 入口点

* ENTRYPOINT 的目的和 CMD 一样，都是在指定容器启动程序及参数
* 当指定了 ENTRYPOINT 后， CMD 的含义就发生了改变，不再是直接的运行其命令，而是将CMD 的内容作为参数传给 ENTRYPOINT 指令
* 切换权限，需要预处理时，一般会使用

### ENV 设置环境变量

* ENV <key> <value>
* ENV <key1>=<value1> <key2>=<value2>...
* 无论是后面的其它指令，如 RUN ，还是运行时的应用，都可以直接使用这里定义的环境变量
* 对含有空格的值用双引号括起来的办法
* 有点相当于一个 **全局变量**
* 下列指令可以支持环境变量展开：ADD 、 COPY 、 ENV 、 EXPOSE 、 LABEL 、 USER 、 WORKDIR 、 VOLUME 、 STOPSIGNAL 、 ONBUILD 

### ARG 构建参数
* 格式： ARG <参数名>[=<默认值>]
* 构建参数和 ENV 的效果一样，都是设置环境变量。所不同的是， ARG 所设置环境变量，在将来容器运行时是不会存在这些环境变量的。但是不要因此就使用 ARG 保存密码之类的信息，因为 docker history 还是可以看到所有值的
* Dockerfile 中的 ARG 指令是定义参数名称，以及定义其默认值。该默认值可以在构建命令 docker build 中用 --build-arg <参数名>=<值> 来覆盖

### VOLUME 定义匿名卷

* VOLUME ["<路径1>", "<路径2>"...]
* VOLUME <路径>
* 为了防止运行时用户忘记将动态文件所保存目录挂载为卷，在Dockerfile 中，我们可以事先指定某些目录挂载为匿名卷，这样在运行时如果用户不指定挂载，其应用也可以正常运行，不会向容器存储层写入大量数据。

```VOLUME /data```

这里的 /data 目录就会在运行时自动挂载为匿名卷，任何向 /data 中写入的信息都不会记录进容器存储层，从而保证了容器存储层的无状态化。当然，运行时可以覆盖这个挂载设
置。比如：

```docker run -d -v mydata:/data xxxx```

在这行命令中，就使用了 mydata 这个命名卷挂载到了 /data 这个位置，替代了 Dockerfile 中定义的匿名卷的挂载配置 

### EXPOSE 声明端口

* 格式为 EXPOSE <端口1> [<端口2>...] 

    **EXPOSE 指令是声明运行时容器提供服务端口，这只是一个声明，在运行时并不会因为这个声明应用就会开启这个端口的服务。** 在 Dockerfile 中写入这样的声明有两个好处：

    1. 帮助镜像使用者理解这个镜像服务的守护端口，以方便配置映射
    2. 在运行时使用随机端口映射时，也就是 docker run -P 时，会自动随机映射 EXPOSE 的端口
    
要将 EXPOSE 和在运行时使用 -p <宿主端口>:<容器端口> 区分开来。 -p ，是映射宿主端口和容器端口，换句话说，就是将容器的对应端口服务公开给外界访问，而 EXPOSE 仅仅是声明容器打算使用什么端口而已，并不会自动在宿主进行端口映射

### WORKDIR 指定工作目录

* 格式为 WORKDIR <工作目录路径> 

使用 WORKDIR 指令可以来指定工作目录（或者称为当前目录） ，以后各层的当前目录就被改为指定的目录，如该目录不存在， WORKDIR 会帮你建立目录

* 如果需要改变以后各层的工作目录的位置，那么应该使用 WORKDIR 指令

### USER 指定当前用户

* 格式： USER <用户名>

USER 指令和 WORKDIR 相似，都是改变环境状态并**影响以后的层。** WORKDIR 是改变工作目录， USER 则是改变之后层的执行 RUN , CMD 以及 ENTRYPOINT 这类命令的身份。
当然，和 WORKDIR 一样， USER 只是帮助你切换到指定用户而已，这个用户必须是事先建立好的，否则无法切换

### HEALTHCHECK 健康检查

* 格式：
    * HEALTHCHECK [选项] CMD <命令> ：设置检查容器健康状况的命令
    * HEALTHCHECK NONE ：如果基础镜像有健康检查指令，使用这行可以屏蔽掉其健康检查指令

HEALTHCHECK 指令是告诉 Docker 应该如何进行判断容器的状态是否正常

### ONBUILD 为他人做嫁衣裳

* 格式： ONBUILD <其它指令> 

ONBUILD 是一个特殊的指令，它后面跟的是其它指令，比如 RUN , COPY 等，而这些指令，在当前镜像构建时并不会被执行。只有当以当前镜像为基础镜像，去构建下一级镜像的时候才会被执行

### 保存镜像

比如我们希望保存这个 alpine 镜像

保存镜像的命令为：
```$ docker image save alpine | gzip > alpine-latest.tar.gz```

然后我们将 alpine-latest.tar.gz 文件复制到了到了另一个机器上，可以用下面这个命令加
载镜像：

```$ docker load -i alpine-latest.tar.gz```

如果我们结合这两个命令以及 ssh 甚至 pv 的话，利用 Linux 强大的管道，我们可以写一个命令完成从一个机器将镜像迁移到另一个机器，并且带进度条的功能：

```docker image save <镜像名> | bzip2 | pv | ssh <用户名>@<主机名> 'cat | docker load'```

### 删除本地镜像

如果要删除本地的镜像，可以使用 docker rmi 命令，其格式为：

```docker rmi [选项] <镜像1> [<镜像2> ...]```

注意 docker rm 命令是删除容器，不要混淆。用 ID、镜像名、摘要删除镜像,其中， <镜像> 可以是 镜像短 ID 、 镜像长 ID 、 镜像名 或者 镜像摘要 

镜像的唯一标识是其 ID 和摘要，而一个镜像可以有多个标签。

并非所有的 `docker rmi` 都会产生删除镜像的行为，有可能仅仅是取消了某个标签而已。

* 删除镜像的标准方式
```
docker images --digests
```

某个其它镜像正依赖于当前镜像的某一层。这种情况，依旧不会触发删除该层的行为。直到没有任何层依赖当前层时，才会真实的删除当前层

如果有用这个镜像启动的容器存在（即使容器没有运行） ，那么同样不可以删除这个镜像。容器是以镜像为基础，再加一层容器存储层，组成这样的多层存储结构去运行的。因此该镜像如果被这个容器所依
赖的，那么删除必然会导致故障。如果这些容器是不需要的，应该先将它们删除，然后再来删除镜像。

* docker images -q 来配合使用 `docker rmi` ，这样可以成批的删除希望删除的镜像
* `docker image rm` 推荐方式


